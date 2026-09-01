"""타깃 정의 실험. 검증 구간에서만 돌린다.

지금까지 테스트 구간을 여러 모델 비교에 반복해 썼다. 시계열은 자기상관 때문에
한 줄기로 개선을 반복하면 과적합에 빠지기 쉬우므로, 여기서는 검증 구간만 보고
테스트 구간은 건드리지 않는다.

세 가지 타깃을 비교한다.
  smoothed : 평활한 QoE의 절대값 (현재 방식)
  raw      : 평활 전 원계열의 절대값 (표시용 평활과 예측용 계열 분리)
  delta    : 평활한 QoE의 변화량. 기준선의 예측이 항상 0이 된다
"""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

from qoe_features import FEATURES, GROUP_KEY, prepare
from transition_metrics import to_class, transition_scores

SEED = 42
DEFAULT_CSV = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "backend", "db", "dataset_plus.csv",
)


def three_way_split(df, val_ratio=0.2, test_ratio=0.2):
    """AP별로 앞 60% 학습, 다음 20% 검증, 마지막 20% 테스트."""
    parts = {"train": [], "val": [], "test": []}
    for _, g in df.groupby(GROUP_KEY, sort=False):
        n = len(g)
        a = int(n * (1 - val_ratio - test_ratio))
        b = int(n * (1 - test_ratio))
        parts["train"].extend(g.index[:a])
        parts["val"].extend(g.index[a:b])
        parts["test"].extend(g.index[b:])
    return {k: df.loc[sorted(v)] for k, v in parts.items()}


def build_target(df, kind, horizon, group_key=GROUP_KEY):
    """타깃 정의를 바꿔 붙인다. 원계열은 평활 전 bad_score를 쓴다."""
    if kind == "raw":
        return df.groupby(group_key)["bad_score"].shift(-horizon), df["bad_score"]
    future = df.groupby(group_key)["QoE_index"].shift(-horizon)
    if kind == "delta":
        return future - df["QoE_index"], df["QoE_index"]
    return future, df["QoE_index"]


def run(df, kind, horizon):
    target, current = build_target(df, kind, horizon)
    work = df.copy()
    work["_target"] = target
    work["_current"] = current
    work = work.dropna(subset=["_target"])

    splits = three_way_split(work)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(splits["train"][FEATURES].values)
    X_va = scaler.transform(splits["val"][FEATURES].values)
    y_tr = splits["train"]["_target"].values
    y_va = splits["val"]["_target"].values
    cur_va = splits["val"]["_current"].values

    model = lgb.LGBMRegressor(
        n_estimators=400, learning_rate=0.05, random_state=SEED, verbose=-1
    )
    model.fit(X_tr, y_tr)
    pred = model.predict(X_va)

    # 기준선: 절대값 타깃이면 현재값, 변화량 타깃이면 0
    baseline = np.zeros_like(y_va) if kind == "delta" else cur_va

    # 어떤 타깃이든 "예측된 미래 상태"로 되돌려 같은 자로 잰다
    if kind == "delta":
        fut_pred, fut_true = cur_va + pred, cur_va + y_va
        ref = splits["train"]["_current"].values
    else:
        fut_pred, fut_true = pred, y_va
        ref = y_tr

    q1, q2 = np.quantile(ref, 0.33), np.quantile(ref, 0.66)
    m = transition_scores(
        to_class(fut_true, q1, q2), to_class(fut_pred, q1, q2), to_class(cur_va, q1, q2)
    )

    mae, mae_base = mean_absolute_error(y_va, pred), mean_absolute_error(y_va, baseline)
    return {
        "target": kind,
        "n_val": len(y_va),
        "MAE": mae,
        "MAE_base": mae_base,
        "improve": mae_base - mae,
        "tr_recall": m["recall"],
        "tr_precision": m["precision"],
        "false_alarm": m["false_alarm_rate"],
    }


def main(args):
    df = prepare(args.csv, horizon=args.horizon)
    out = pd.DataFrame(
        [run(df, k, args.horizon) for k in ("smoothed", "raw", "delta")]
    ).set_index("target")
    print(f"[검증 구간 평가] horizon={args.horizon}, 테스트 구간은 사용하지 않음\n")
    print(out.to_string(float_format=lambda v: f"{v:.4f}"))
    print("\nimprove가 양수면 모델이 기준선보다 나은 것이다.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=DEFAULT_CSV)
    parser.add_argument("--horizon", type=int, default=1)
    args = parser.parse_args()
    main(args)
