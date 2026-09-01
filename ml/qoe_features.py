"""QoE 지표 정의와 피처 생성.

학습 스크립트(train_models.py)와 서빙 코드(backend/app.py)가 같은 정의를
쓰도록 QoE 산식을 한 곳에 모아둔 모듈이다.
"""

import numpy as np
import pandas as pd

# 정규화 상한. 도서관 WiFi 로그의 상위 구간과 체감 한계를 함께 보고 잡았다.
MAX_PING, MAX_JITTER, MAX_LOSS = 500.0, 100.0, 20.0
MAX_SPEED, MAX_CLIENTS = 500.0, 100.0

# v1: 수상 당시의 지표. 원인 변수(RSSI, 접속자 수)가 지수 안에 섞여 있고,
# 손실률은 97.3%의 행에서 0이라 0.20을 줘도 지수를 거의 움직이지 못한다.
WEIGHTS_V1 = {
    "norm_ping": 0.25,
    "norm_jitter": 0.15,
    "norm_loss": 0.20,
    "norm_speed": 0.25,
    "norm_rssi": 0.10,
    "norm_clients": 0.05,
}

# v2: 체감 변수만 남긴 지표.
#
# QoE는 이용자가 무엇을 겪었는가를 재는 값이다. RSSI와 접속자 수는 그 체감을
# 만들어내는 원인이지 체감 자체가 아니다. 검증해보니 두 변수가 체감 지수를
# 설명하는 정도는 R^2 0.065에 그쳐, 지수에 넣으면 체감과 거의 무관한 성분을
# 15% 섞는 셈이 된다. 그래서 원인 변수는 지수에서 빼고 예측 피처로만 쓴다.
#
# 손실률도 뺐다. 2.7%의 행에서만 0보다 크므로 연속 지수의 항목으로는 죽은
# 변수다. 대신 loss_event 플래그로 따로 다룬다.
#
# 가중치는 의도한 비중(핑 0.25, 지터 0.15, 속도 0.25)이 실효 기여도로
# 실현되도록 각 항목의 표준편차로 나눠 역보정한 값이다.
WEIGHTS_V2 = {
    "norm_ping": 0.365,
    "norm_jitter": 0.113,
    "norm_speed": 0.522,
}

# 지수에서 뺐지만 예측에는 쓰는 원인 변수
CAUSE_FEATURES = ["RSSI", "client", "norm_rssi", "norm_clients"]

WEIGHTS = WEIGHTS_V1  # 하위 호환. 새 코드는 qoe_weights()를 쓴다


def qoe_weights(version="v2"):
    return dict(WEIGHTS_V2 if version == "v2" else WEIGHTS_V1)

# 수집 간격은 AP당 중앙값 1분이다. span=12는 약 12분 평활에 해당한다.
EWM_SPAN = 12

TIMESTEPS_DEFAULT = 6  # 12분치 시퀀스

GROUP_KEY = "ap_code"  # AP 4대가 한 파일에 섞여 있어 반드시 나눠서 처리한다

NUMERIC_COLS = [
    "ping_ms", "ping_jitter_ms", "packet_loss_rate",
    "download_Mbps", "upload_Mbps", "RSSI", "client",
]

FEATURES = [
    "ping_ms", "ping_jitter_ms", "packet_loss_rate",
    "log_download", "log_upload", "RSSI", "client",
    "norm_ping", "norm_speed", "norm_loss",
    "ping_ms_diff", "download_Mbps_diff",
    # 원인 변수는 지수에서 뺀 대신 여기에 남는다
    "norm_rssi", "norm_clients", "loss_event",
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


def add_qoe_index(df, group_key=GROUP_KEY, version="v2"):
    """가중합 저하 점수를 만든 뒤 EWM으로 평활해 QoE_index를 붙인다.

    평활은 AP별로 따로 건다. 한 파일에 여러 AP가 시간순으로 섞여 있어서
    전체에 한 번에 걸면 다른 층의 값이 서로 스며든다.
    """
    df["norm_ping"] = log_norm_bad(df["ping_ms"], MAX_PING)
    df["norm_jitter"] = log_norm_bad(df["ping_jitter_ms"], MAX_JITTER)
    df["norm_loss"] = log_norm_bad(df["packet_loss_rate"], MAX_LOSS)
    df["norm_speed"] = log_norm_good(df["download_Mbps"], MAX_SPEED)
    df["norm_rssi"] = clamp((df["RSSI"] + 40) / -50.0)
    df["norm_clients"] = log_norm_bad(df["client"], MAX_CLIENTS)

    weights = qoe_weights(version)
    bad_score = np.zeros(len(df))
    for col, w in weights.items():
        bad_score += df[col].values * w

    df["bad_score"] = bad_score
    if group_key in df.columns:
        smoothed = df.groupby(group_key)["bad_score"].transform(
            lambda s: s.ewm(span=EWM_SPAN, adjust=False).mean()
        )
    else:
        smoothed = df["bad_score"].ewm(span=EWM_SPAN, adjust=False).mean()

    df["QoE_index"] = clamp(smoothed)
    return df


def add_features(df, group_key=GROUP_KEY):
    """차분도 AP별로 계산한다. 전체에 걸면 AP 경계에서 엉뚱한 값이 나온다."""
    # 손실은 드물게 일어나는 사건이라 연속값보다 발생 여부가 정보량이 크다.
    df["loss_event"] = (df["packet_loss_rate"] > 0).astype(float)
    df["log_download"] = np.log1p(df["download_Mbps"])
    df["log_upload"] = np.log1p(df["upload_Mbps"])
    for col in ("ping_ms", "download_Mbps"):
        if group_key in df.columns:
            df[f"{col}_diff"] = df.groupby(group_key)[col].diff().fillna(0)
        else:
            df[f"{col}_diff"] = df[col].diff().fillna(0)
    return df


def add_target(df, horizon=1, group_key=GROUP_KEY):
    """horizon 스텝 뒤의 QoE를 타깃으로 둔다. 시프트도 AP별로 한다."""
    if group_key in df.columns:
        df["QoE_index_future"] = df.groupby(group_key)["QoE_index"].shift(-horizon)
    else:
        df["QoE_index_future"] = df["QoE_index"].shift(-horizon)
    return df.dropna(subset=["QoE_index_future"]).reset_index(drop=True)


def prepare(path, horizon=1, group_key=GROUP_KEY, sort=True, version="v2"):
    """AP별로 정렬한 뒤 지표·피처·타깃을 만든다.

    AP별로 묶어 두면 시퀀스를 자를 때도 한 AP 안에서만 잘린다.
    """
    df = load_dataset(path)
    if sort and group_key in df.columns and "datetime" in df.columns:
        df["_dt"] = pd.to_datetime(df["datetime"], errors="coerce")
        df = df.sort_values([group_key, "_dt"]).drop(columns="_dt").reset_index(drop=True)
    df = add_qoe_index(df, group_key, version=version)
    df = add_features(df, group_key)
    return add_target(df, horizon=horizon, group_key=group_key)


def make_sequences(X, y, timesteps, groups=None):
    """시퀀스를 자른다. groups를 주면 AP 경계를 넘는 시퀀스는 버린다."""
    xs, ys, keep = [], [], []
    for i in range(timesteps, len(X)):
        if groups is not None and len(set(groups[i - timesteps:i + 1])) > 1:
            continue
        xs.append(X[i - timesteps:i])
        ys.append(y[i])
        keep.append(i)
    return np.array(xs), np.array(ys), np.array(keep)