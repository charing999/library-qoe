"""QoE 가중치 감도 분석.

여섯 개 가중치는 도메인 직관으로 정한 값이다. 근거를 더 대야 하는지 판단하려면
먼저 "흔들면 결과가 바뀌는가"를 봐야 한다. 안 바뀌면 가중치 논쟁 자체가
무의미하고, 크게 바뀌면 근거를 대야 한다.

두 가지를 잰다.
  1. 가중치를 무작위로 흔들었을 때 3단계 라벨이 얼마나 유지되는가
  2. 가중치 하나씩 0으로 만들었을 때 (제거) 라벨이 얼마나 바뀌는가
"""

import argparse
import os

import numpy as np
import pandas as pd

from qoe_features import (
    EWM_SPAN, GROUP_KEY, WEIGHTS, clamp, load_dataset, log_norm_bad,
    log_norm_good, MAX_CLIENTS, MAX_JITTER, MAX_LOSS, MAX_PING, MAX_SPEED,
)

DEFAULT_CSV = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "backend", "db", "dataset_plus.csv",
)


def norm_components(df):
    """가중치와 무관한 정규화 성분만 미리 만들어 둔다."""
    return pd.DataFrame({
        "norm_ping": log_norm_bad(df["ping_ms"], MAX_PING),
        "norm_jitter": log_norm_bad(df["ping_jitter_ms"], MAX_JITTER),
        "norm_loss": log_norm_bad(df["packet_loss_rate"], MAX_LOSS),
        "norm_speed": log_norm_good(df["download_Mbps"], MAX_SPEED),
        "norm_rssi": clamp((df["RSSI"] + 40) / -50.0),
        "norm_clients": log_norm_bad(df["client"], MAX_CLIENTS),
    })


def qoe_from_weights(components, weights, groups):
    """가중치를 받아 QoE 계열을 만든다. 평활은 AP별로 건다."""
    bad = np.zeros(len(components))
    for col, w in weights.items():
        bad += components[col].values * w

    s = pd.Series(bad)
    if groups is not None:
        s = s.groupby(groups).transform(
            lambda x: x.ewm(span=EWM_SPAN, adjust=False).mean()
        )
    else:
        s = s.ewm(span=EWM_SPAN, adjust=False).mean()
    return clamp(s.values)


def to_class(values):
    """분위수 기준이므로 가중치를 바꾸면 임계값도 따라 움직인다."""
    q1, q2 = np.quantile(values, 0.33), np.quantile(values, 0.66)
    return np.where(values <= q1, 0, np.where(values <= q2, 1, 2))


def perturb(weights, rng, scale):
    """각 가중치에 곱셈 잡음을 주고 합이 1이 되도록 다시 정규화한다."""
    keys = list(weights)
    vals = np.array([weights[k] for k in keys])
    noisy = vals * np.exp(rng.normal(0, scale, len(vals)))
    noisy /= noisy.sum()
    return dict(zip(keys, noisy))


def main(args):
    df = load_dataset(args.csv)
    groups = df[GROUP_KEY].values if GROUP_KEY in df.columns else None
    comp = norm_components(df)

    base_qoe = qoe_from_weights(comp, WEIGHTS, groups)
    base_cls = to_class(base_qoe)

    rng = np.random.default_rng(42)

    print(f"샘플 {len(df)}건, 시행 {args.trials}회")
    print("\n[1] 가중치를 무작위로 흔들었을 때 3단계 라벨 유지율")
    print(f"{'교란 강도':<12} | {'라벨 일치율':<20} | {'QoE 상관':<10}")
    print("-" * 50)
    for scale in (0.1, 0.25, 0.5):
        agree, corr = [], []
        for _ in range(args.trials):
            w = perturb(WEIGHTS, rng, scale)
            q = qoe_from_weights(comp, w, groups)
            agree.append(np.mean(to_class(q) == base_cls))
            corr.append(np.corrcoef(q, base_qoe)[0, 1])
        print(f"sigma={scale:<6.2f} | {np.mean(agree):.4f} "
              f"(최저 {np.min(agree):.4f}) | {np.mean(corr):.4f}")

    print("\n[2] 가중치를 하나씩 제거했을 때")
    print(f"{'제거한 항목':<15} | {'원래 비중':<9} | {'라벨 일치율':<10}")
    print("-" * 45)
    rows = []
    for drop in WEIGHTS:
        w = {k: v for k, v in WEIGHTS.items() if k != drop}
        total = sum(w.values())
        w = {k: v / total for k, v in w.items()}
        q = qoe_from_weights(comp, w, groups)
        a = np.mean(to_class(q) == base_cls)
        rows.append((drop, WEIGHTS[drop], a))
    for drop, orig, a in sorted(rows, key=lambda r: r[2]):
        print(f"{drop:<15} | {orig:<9.2f} | {a:.4f}")

    print("\n[3] 명목 가중치 vs 실효 기여도")
    # 가중치가 커도 그 지표가 거의 안 변하면 QoE에 기여하지 못한다.
    eff = {k: WEIGHTS[k] * comp[k].std() for k in WEIGHTS}
    total = sum(eff.values())
    print(f"{'항목':<15} | {'명목':<6} | {'실효':<6} | {'0인 비율':<8}")
    print("-" * 45)
    src = {"norm_ping": "ping_ms", "norm_jitter": "ping_jitter_ms",
           "norm_loss": "packet_loss_rate", "norm_speed": "download_Mbps",
           "norm_rssi": "RSSI", "norm_clients": "client"}
    for k, v in sorted(eff.items(), key=lambda x: -x[1]):
        zero = (df[src[k]] == 0).mean()
        print(f"{k:<15} | {WEIGHTS[k]:<6.2f} | {v / total:<6.3f} | {zero:.1%}")

    print("\n[4] 균등 가중치와 비교")
    even = {k: 1 / len(WEIGHTS) for k in WEIGHTS}
    q = qoe_from_weights(comp, even, groups)
    print(f"라벨 일치율 {np.mean(to_class(q) == base_cls):.4f}, "
          f"QoE 상관 {np.corrcoef(q, base_qoe)[0, 1]:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=DEFAULT_CSV)
    parser.add_argument("--trials", type=int, default=200)
    args = parser.parse_args()
    main(args)
