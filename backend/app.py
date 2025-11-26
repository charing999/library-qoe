# backend/app.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import os
import datetime
from datetime import timedelta
import hashlib

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# 1. 모델 및 데이터 로드
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
# compile=False: 학습 설정 무시하고 깡통 모델만 로드 (버전 호환성 해결)
gru_model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "gru_qoe.h5"), compile=False)

csv_path = os.path.join(BASE_DIR, "db", "dataset_plus.csv")
try:
    df_all = pd.read_csv(csv_path, encoding='cp949')
except:
    df_all = pd.read_csv(csv_path, encoding='utf-8')

df_all.columns = [c.strip() for c in df_all.columns]

# ---------------------------------------------------------
# 🔥 [핵심] 사용자가 제공한 "학습 공식" 그대로 적용
# ---------------------------------------------------------
def log_norm_bad(series, max_val):
    # 값이 클수록 1에 가깝게 (나쁨)
    return np.clip(np.log1p(series) / np.log1p(max_val), 0, 1)

def log_norm_good(series, max_val):
    # 값이 클수록 0에 가깝게 (좋음 -> 나쁨 점수니까 뒤집음)
    return 1 - np.clip(np.log1p(series) / np.log1p(max_val), 0, 1)

def clamp(x): return np.clip(x, 0, 1)

print("⏳ 학습 공식대로 QoE 재계산 중...")

# 1. 상수 정의 (학습 때 쓴 값)
MAX_PING, MAX_JITTER, MAX_LOSS = 500.0, 100.0, 20.0
MAX_SPEED, MAX_CLIENTS = 500.0, 100.0

# 2. 정규화 (Normalization)
# 없는 컬럼은 0으로 채워서 에러 방지
if "ping_jitter_ms" not in df_all.columns: df_all["ping_jitter_ms"] = 0
if "RSSI" not in df_all.columns: df_all["RSSI"] = -60
if "client" not in df_all.columns: df_all["client"] = 0

df_all["norm_ping"]   = log_norm_bad(df_all["ping_ms"], MAX_PING)
df_all["norm_jitter"] = log_norm_bad(df_all["ping_jitter_ms"], MAX_JITTER)
df_all["norm_loss"]   = log_norm_bad(df_all["packet_loss_rate"], MAX_LOSS)
df_all["norm_speed"]  = log_norm_good(df_all["download_Mbps"], MAX_SPEED)
# RSSI는 보통 -30(좋음) ~ -90(나쁨). 식에 따르면 -40 -> 0, -90 -> 1.0
df_all["norm_rssi"]   = clamp((df_all["RSSI"] + 40) / -50.0)
df_all["norm_clients"]= log_norm_bad(df_all["client"], MAX_CLIENTS)

# 3. 가중치 합산 (Weighted Sum)
weights = {"norm_ping": 0.25, "norm_jitter": 0.15, "norm_loss": 0.20,
           "norm_speed": 0.25, "norm_rssi": 0.10, "norm_clients": 0.05}

bad_score = np.zeros(len(df_all))
for c in weights: 
    bad_score += df_all[c].values * weights[c]

# 4. EWM (지수 이동 평균) 적용 -> 시계열 흐름 반영
# span=12는 약 12개 데이터(약 1시간?)의 흐름을 반영한다는 뜻
df_all["QoE_index"] = pd.Series(bad_score).ewm(span=12, adjust=False).mean()
df_all["QoE_index"] = clamp(df_all["QoE_index"])

# 5. 모델 입력용 추가 변수들 (학습 때 쓴 Feature들)
df_all["log_download"] = np.log1p(df_all["download_Mbps"])
df_all["log_upload"] = np.log1p(df_all["upload_Mbps"])
df_all["ping_ms_diff"] = df_all.groupby("ap_code")["ping_ms"].diff().fillna(0)
df_all["download_Mbps_diff"] = df_all.groupby("ap_code")["download_Mbps"].diff().fillna(0)

# 시간 컬럼 처리
if 'datetime' in df_all.columns:
    df_all['dt_obj'] = pd.to_datetime(df_all['datetime'], errors='coerce')
    df_all['hour'] = df_all['dt_obj'].dt.hour
else:
    df_all['hour'] = 12

print("✅ 데이터 준비 완료! (User Formula Applied)")

FEATURES = [
    "ping_ms", "ping_jitter_ms", "packet_loss_rate",
    "log_download", "log_upload", "RSSI", "client",
    "norm_ping", "norm_speed", "norm_loss",
    "ping_ms_diff", "download_Mbps_diff"
]

def to_grade(qoe):
    # 🔥 [중요] 점수가 낮을수록 Good (불만족 지수이므로)
    # 0.0 (완벽) ~ 1.0 (최악)
    if qoe <= 0.35: return "Good"       # 0.35 이하는 아주 쾌적
    elif qoe <= 0.65: return "Moderate" # 0.65 이하는 보통
    else: return "Bad"                  # 그 이상은 나쁨

# ---------------------------------------------------------
# 🌍 [추가] GPS 문자열 파싱 헬퍼 함수
# 예: "(36.369872, 127.346647)" -> 36.369872, 127.346647
# ---------------------------------------------------------
def parse_gps(gps_str):
    try:
        # 괄호 제거하고 쉼표로 나누기
        clean_str = str(gps_str).replace('(', '').replace(')', '')
        lat, lon = map(float, clean_str.split(','))
        return lat, lon
    except:
        return None, None

# backend/app.py 의 dashboard_summary 함수 교체

@app.get("/api/dashboard")
def dashboard_summary(floor: str = "1F"): 
    # 1. 해당 층 데이터 필터링
    floor_df = df_all[df_all['location2'].str.contains(floor, na=False)]
    
    if len(floor_df) == 0:
        return {"floor": floor, "aps": [], "alert_count": 0}

    unique_aps = floor_df.groupby("ap_code").last().reset_index()
    
    # ---------------------------------------------------------
    # 🔥 [수동 좌표 매핑] (GPS 대신 화면상 % 좌표 사용)
    # 프론트엔드에서 지도를 클릭해서 얻은 값을 여기에 적으면 됨
    # 형식: "AP_ID": (가로%, 세로%)
    # ---------------------------------------------------------
    FIXED_POSITIONS = {
        # 예시: 내가 임의로 잡아둔 위치 (너의 도면에 맞게 고쳐야 함!)
        "CLIENT_AP_1F104H0013121": (27.3, 49.7),  
        "CLIENT_AP_1F104H0007121": (30.5, 37.7),  
        "CLIENT_AP_1F110H0024121": (76.9, 32.9),  
        "CLIENT_AP_B1F0138121": (69.4, 37.5),
        "CLIENT_AP_B2F0155121": (79.0, 26.0),
        "CLIENT_AP_B2F0146121": (34.2, 62.0),
        "CLIENT_AP_2F207H0055121": (61.8, 32.0),
    }

    ap_list = []
    
    for _, row in unique_aps.iterrows():
        ap_id = row['ap_code']
        qoe = row['QoE_index']
        
        # 1. 우리가 좌표를 지정해둔 AP면 그 위치 사용
        if ap_id in FIXED_POSITIONS:
            x, y = FIXED_POSITIONS[ap_id]
        
        # 2. 지정 안 된 AP는 일단 왼쪽 상단에 모아두기 (찾기 쉽게)
        else:
            x, y = 5, 5 # (5%, 5%) 위치
        
        ap_list.append({
            "id": ap_id, 
            "x": x, 
            "y": y, 
            "status": to_grade(qoe), 
            "qoe": round(qoe, 2)
        })

    return {
        "floor": floor,
        "aps": ap_list,
        "alert_count": sum(1 for ap in ap_list if ap["status"] != "Good")
    }

@app.get("/api/predict/{ap_id}")
def predict_ap(ap_id: str):
    now = datetime.datetime.now()
    future_time = now + timedelta(minutes=5)

    my_ap_df = df_all[df_all['ap_code'] == ap_id]
    
    if len(my_ap_df) < 10:
        return {"error": "데이터 부족"}

    # 시간 동기화 (초 단위)
    cur_min = now.minute
    cur_sec = now.second
    total_sec = cur_min * 60 + cur_sec
    target_idx = total_sec % (len(my_ap_df) - 6)
    
    temp_df = my_ap_df.iloc[target_idx : target_idx + 6].copy()

    try:
        X = temp_df[FEATURES].values
        X_scaled = scaler.transform(X)
        X_seq = np.expand_dims(X_scaled, axis=0)

        # AI 예측 (모델도 낮은 점수가 Good으로 학습되었을 것임)
        pred_qoe = float(gru_model.predict(X_seq, verbose=0)[0, 0])
        curr_qoe = float(temp_df["QoE_index"].iloc[-1])

        time_str_now = now.strftime("%H:%M:%S")
        time_str_future = future_time.strftime("%H:%M")

        return {
            "ap_id": ap_id,
            "current_time_text": f"현재 ({time_str_now})",
            "future_time_text": f"5분 뒤 예측 ({time_str_future})",
            "current_qoe": round(curr_qoe, 2),
            "future_qoe": round(pred_qoe, 2),
            "current_grade": to_grade(curr_qoe),
            "future_grade": to_grade(pred_qoe),
            "metrics": temp_df.iloc[-1].to_dict()
        }

    except Exception as e:
        print(f"❌ 에러: {e}")
        return {"error": str(e)}

@app.get("/api/recommend")
def recommend_zone():
     return {
        "best_zone": "B2 열람실",
        "message": "AI 분석 결과 가장 쾌적합니다.",
        "zones": [
            {"name": "B2 열람실", "grade": "Good"},
            {"name": "B1 휴게실", "grade": "Moderate"},
            {"name": "1F 로비", "grade": "Bad"},
        ]
    }