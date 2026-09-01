"""전이 검출 관점의 평가.

전체 정확도는 상태가 잘 안 바뀌는 계열에서 항상 높게 나온다. 정작 서비스가
필요로 하는 건 "나빠지기 전에 알렸는가"이고, 그 순간은 드물어서 전체
정확도에 거의 잡히지 않는다. 그래서 전이 구간만 떼어 따로 잰다.

Persistence는 정의상 전이를 예측하지 못하므로 이 지표에서는 0점이다.
비교가 성립하는 판을 만드는 것이 이 모듈의 목적이다.
"""

import numpy as np


def to_class(values, t1, t2):
    """0 = Good, 1 = Moderate, 2 = Bad."""
    return np.where(values <= t1, 0, np.where(values <= t2, 1, 2))


def transition_scores(y_true_cls, y_pred_cls, prev_cls):
    """전이 구간에 한정한 재현율·정밀도·헛경보율.

    prev_cls는 예측 시점의 현재 상태다. 실제로 상태가 바뀐 구간을 양성으로 본다.
    """
    actual_change = y_true_cls != prev_cls
    pred_change = y_pred_cls != prev_cls

    tp = int(np.sum(actual_change & pred_change & (y_pred_cls == y_true_cls)))
    detected = int(np.sum(actual_change & pred_change))
    n_actual = int(np.sum(actual_change))
    n_pred = int(np.sum(pred_change))

    # 안 바뀌는데 바뀐다고 한 경우
    false_alarm = int(np.sum(~actual_change & pred_change))
    n_stable = int(np.sum(~actual_change))

    return {
        "n_transitions": n_actual,
        "transition_rate": n_actual / len(y_true_cls) if len(y_true_cls) else 0.0,
        # 전이를 전이라고 맞힌 비율 (방향 무관)
        "recall": detected / n_actual if n_actual else 0.0,
        # 전이라고 한 것 중 실제 전이였던 비율
        "precision": detected / n_pred if n_pred else 0.0,
        # 전이를 맞히면서 도착 상태까지 맞힌 비율
        "recall_exact": tp / n_actual if n_actual else 0.0,
        "false_alarm_rate": false_alarm / n_stable if n_stable else 0.0,
    }


def degradation_scores(y_true_cls, y_pred_cls, prev_cls):
    """악화 전이(Good/Moderate -> 더 나쁨)만 본다. 서비스가 실제로 알려야 할 사건이다."""
    worsen_true = y_true_cls > prev_cls
    worsen_pred = y_pred_cls > prev_cls

    tp = int(np.sum(worsen_true & worsen_pred))
    fp = int(np.sum(~worsen_true & worsen_pred))
    fn = int(np.sum(worsen_true & ~worsen_pred))

    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * recall * precision / (recall + precision) if (recall + precision) else 0.0
    return {
        "n_degradations": tp + fn,
        "recall": recall,
        "precision": precision,
        "f1": f1,
    }


def direction_accuracy(y_true, y_pred, current):
    """방향 일치율. 변화가 없는 구간은 제외하고 오르내림만 맞는지 본다."""
    true_delta = y_true - current
    pred_delta = y_pred - current
    moved = np.abs(true_delta) > 1e-9
    if not moved.any():
        return 0.0
    return float(np.mean(np.sign(true_delta[moved]) == np.sign(pred_delta[moved])))


def report(name, y_true, y_pred, current, t1, t2):
    """한 모델의 전이 관점 성적을 dict로 돌려준다."""
    true_cls = to_class(y_true, t1, t2)
    pred_cls = to_class(y_pred, t1, t2)
    prev_cls = to_class(current, t1, t2)

    tr = transition_scores(true_cls, pred_cls, prev_cls)
    dg = degradation_scores(true_cls, pred_cls, prev_cls)
    return {
        "model": name,
        "transition_recall": tr["recall"],
        "transition_precision": tr["precision"],
        "false_alarm_rate": tr["false_alarm_rate"],
        "degradation_recall": dg["recall"],
        "degradation_precision": dg["precision"],
        "degradation_f1": dg["f1"],
        "direction_acc": direction_accuracy(y_true, y_pred, current),
    }
