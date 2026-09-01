"""Persistence 기준선.

"5분 뒤 QoE는 지금 QoE와 같다"고 답하는 모델이다. GRU가 이보다 얼마나
나은지 밝히지 않으면 회귀 지표만으로는 아무것도 증명하지 못한다.
"""

import argparse
import os

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from qoe_features import TIMESTEPS_DEFAULT, prepare

DEFAULT_CSV = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "backend", "db", "dataset_plus.csv",
)


def evaluate(csv_path, timesteps=TIMESTEPS_DEFAULT, test_ratio=0.2, horizon=1):
    df = prepare(csv_path, horizon=horizon)

    split_idx = int(len(df) * (1 - test_ratio))
    qoe = df["QoE_index"].values
    y = df["QoE_index_future"].values

    # 학습 스크립트와 동일한 분할·시퀀스 오프셋을 써서 테스트 구간을 맞춘다.
    start = split_idx + timesteps
    y_true = y[start:]
    y_pred = qoe[start:]

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    return {"n_test": len(y_true), "MSE": mse, "RMSE": float(np.sqrt(mse)), "MAE": mae}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=DEFAULT_CSV)
    parser.add_argument("--timesteps", type=int, default=TIMESTEPS_DEFAULT)
    parser.add_argument("--horizon", type=int, default=1, help="몇 스텝(5분) 뒤를 예측할지")
    args = parser.parse_args()

    scores = evaluate(args.csv, timesteps=args.timesteps, horizon=args.horizon)
    print(f"[Persistence baseline] QoE(t+{args.horizon}) = QoE(t)")
    for k, v in scores.items():
        print(f"  {k:<6} {v:.6f}" if isinstance(v, float) else f"  {k:<6} {v}")
