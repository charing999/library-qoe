"""QoE 지표 정의와 피처 생성.

학습 스크립트(train_models.py)와 서빙 코드(backend/app.py)가 같은 정의를
쓰도록 QoE 산식을 한 곳에 모아둔 모듈이다.
"""

import numpy as np
import pandas as pd

# 정규화 상한. 도서관 WiFi 로그의 상위 구간과 체감 한계를 함께 보고 잡았다.
MAX_PING, MAX_JITTER, MAX_LOSS = 500.0, 100.0, 20.0
MAX_SPEED, MAX_CLIENTS = 500.0, 100.0

# 가중치 합은 1.0. 지연/손실 계열에 0.6을 두어 체감 저하를 먼저 반영한다.
WEIGHTS = {
    "norm_ping": 0.25,
    "norm_jitter": 0.15,
    "norm_loss": 0.20,
    "norm_speed": 0.25,
    "norm_rssi": 0.10,
    "norm_clients": 0.05,
}

EWM_SPAN = 12  # 5분 간격 로그 기준 약 1시간 평활

TIMESTEPS_DEFAULT = 6  # 30분치 시퀀스를 보고 5분 뒤를 예측

NUMERIC_COLS = [
    "ping_ms", "ping_jitter_ms", "packet_loss_rate",
    "download_Mbps", "upload_Mbps", "RSSI", "client",
]

FEATURES = [
    "ping_ms", "ping_jitter_ms", "packet_loss_rate",
    "log_download", "log_upload", "RSSI", "client",
    "norm_ping", "norm_speed", "norm_loss",
    "ping_ms_diff", "download_Mbps_diff",
]


def log_norm_bad(series, max_val):
    """값이 클수록 나쁜 지표를 0~1로. 낮은 구간의 변화를 더 크게 본다."""
    return np.clip(np.log1p(series) / np.log1p(max_val), 0, 1)


def log_norm_good(series, max_val):
    """값이 클수록 좋은 지표를 뒤집어 0~1 저하 점수로 바꾼다."""
    return 1 - np.clip(np.log1p(series) / np.log1p(max_val), 0, 1)


def clamp(x):
    return np.clip(x, 0, 1)


def load_dataset(path):
    """인코딩이 섞여 있는 수집 로그를 순서대로 시도해 읽는다."""
    for enc in ("utf-8-sig", "cp949", "euc-kr", "latin1"):
        try:
            df = pd.read_csv(path, encoding=enc)
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    else:
        raise ValueError(f"CSV 로드 실패: {path}")

    df.columns = [c.strip() for c in df.columns]
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    present = [c for c in NUMERIC_COLS if c in df.columns]
    return df.dropna(subset=present).reset_index(drop=True)


def add_qoe_index(df):
    """가중합 저하 점수를 만든 뒤 EWM으로 평활해 QoE_index를 붙인다."""
    df["norm_ping"] = log_norm_bad(df["ping_ms"], MAX_PING)
    df["norm_jitter"] = log_norm_bad(df["ping_jitter_ms"], MAX_JITTER)
    df["norm_loss"] = log_norm_bad(df["packet_loss_rate"], MAX_LOSS)
    df["norm_speed"] = log_norm_good(df["download_Mbps"], MAX_SPEED)
    df["norm_rssi"] = clamp((df["RSSI"] + 40) / -50.0)
    df["norm_clients"] = log_norm_bad(df["client"], MAX_CLIENTS)

    bad_score = np.zeros(len(df))
    for col, w in WEIGHTS.items():
        bad_score += df[col].values * w

    df["QoE_index"] = clamp(
        pd.Series(bad_score).ewm(span=EWM_SPAN, adjust=False).mean()
    )
    return df


def add_features(df):
    df["log_download"] = np.log1p(df["download_Mbps"])
    df["log_upload"] = np.log1p(df["upload_Mbps"])
    for col in ("ping_ms", "download_Mbps"):
        df[f"{col}_diff"] = df[col].diff().fillna(0)
    return df


def add_target(df, horizon=1):
    """horizon 스텝(수집 간격 5분) 뒤의 QoE를 타깃으로 둔다."""
    df["QoE_index_future"] = df["QoE_index"].shift(-horizon)
    return df.dropna(subset=["QoE_index_future"]).reset_index(drop=True)


def prepare(path, horizon=1):
    df = load_dataset(path)
    df = add_qoe_index(df)
    df = add_features(df)
    return add_target(df, horizon=horizon)


def make_sequences(X, y, timesteps):
    xs, ys = [], []
    for i in range(timesteps, len(X)):
        xs.append(X[i - timesteps:i])
        ys.append(y[i])
    return np.array(xs), np.array(ys)