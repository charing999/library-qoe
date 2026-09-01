"""QoE 예측 모델 학습·비교 스크립트.

GRU / CNN-LSTM / Transformer / LightGBM을 같은 분할과 같은 지표로 비교한다.
Persistence 기준선(baseline.py)도 함께 출력해 개선폭을 확인한다.

    python ml/train_models.py --csv backend/db/dataset_plus.csv
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report,
    mean_absolute_error, mean_squared_error,
)
import lightgbm as lgb

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Add, Conv1D, Dense, Dropout, GlobalAveragePooling1D, GRU, Input,
    LayerNormalization, LSTM, MaxPooling1D, MultiHeadAttention,
)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

from baseline import evaluate as persistence_scores
from qoe_features import FEATURES, TIMESTEPS_DEFAULT, make_sequences, prepare

SEED = 42
EPOCHS = 50
BATCH_SIZE = 32

DEFAULT_CSV = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "backend", "db", "dataset_plus.csv",
)


def set_seeds():
    np.random.seed(SEED)
    tf.random.set_seed(SEED)


def configure_gpu():
    for gpu in tf.config.experimental.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)


def build_gru(input_shape):
    return Sequential([
        GRU(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        GRU(32),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid"),
    ], name="GRU")


def build_cnn_lstm(input_shape):
    return Sequential([
        Conv1D(32, kernel_size=3, activation="relu", padding="same",
               input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        LSTM(64),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid"),
    ], name="CNN-LSTM")


def build_transformer(input_shape):
    inputs = Input(shape=input_shape)

    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(key_dim=32, num_heads=4, dropout=0.3)(x, x)
    x = Dropout(0.3)(x)
    res = Add()([x, inputs])

    x = LayerNormalization(epsilon=1e-6)(res)
    x = Conv1D(64, kernel_size=1, activation="relu")(x)
    x = Dropout(0.3)(x)
    x = Conv1D(input_shape[1], kernel_size=1)(x)
    x = Add()([x, res])

    x = GlobalAveragePooling1D()(x)
    x = Dense(32, activation="relu")(x)
    outputs = Dense(1, activation="sigmoid")(x)
    return Model(inputs, outputs, name="Transformer")


def split_and_scale(df, timesteps, test_ratio=0.2):
    X_all = df[FEATURES].values
    y_all = df["QoE_index_future"].values

    split_idx = int(len(df) * (1 - test_ratio))
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_all[:split_idx])
    X_test = scaler.transform(X_all[split_idx:])
    y_train, y_test = y_all[:split_idx], y_all[split_idx:]

    train_seq = make_sequences(X_train, y_train, timesteps)
    test_seq = make_sequences(X_test, y_test, timesteps)
    return train_seq, test_seq, y_train, scaler


def to_class(arr, t1, t2):
    """Good / Moderate / Bad 3단계. 임계값은 학습 구간 분위수로 잡는다."""
    return np.where(arr <= t1, 0, np.where(arr <= t2, 1, 2))


def main(args):
    set_seeds()
    configure_gpu()

    df = prepare(args.csv, horizon=args.horizon)
    (X_train_seq, y_train_seq), (X_test_seq, y_test_seq), y_train, _ = split_and_scale(
        df, args.timesteps
    )
    print(f"[INFO] Train shape: {X_train_seq.shape}, Test shape: {X_test_seq.shape}")

    input_shape = (args.timesteps, len(FEATURES))
    results, predictions = {}, {}

    for builder in (build_gru, build_cnn_lstm, build_transformer):
        model = builder(input_shape)
        print(f"\nTraining {model.name}...")
        model.compile(optimizer=Adam(learning_rate=0.001), loss="mse", metrics=["mae"])
        early_stop = EarlyStopping(
            monitor="val_loss", patience=7, restore_best_weights=True, verbose=0
        )
        model.fit(
            X_train_seq, y_train_seq,
            epochs=EPOCHS, batch_size=BATCH_SIZE,
            validation_split=0.1, callbacks=[early_stop], verbose=0,
        )
        pred = model.predict(X_test_seq, verbose=0).flatten()
        predictions[model.name] = pred
        results[model.name] = {
            "MSE": mean_squared_error(y_test_seq, pred),
            "MAE": mean_absolute_error(y_test_seq, pred),
        }

    # LightGBM은 3차원 입력을 못 받으므로 (N, timesteps*features)로 펼친다.
    print("\nTraining LightGBM...")
    lgb_model = lgb.LGBMRegressor(
        n_estimators=500, learning_rate=0.05, random_state=SEED, verbose=-1
    )
    lgb_model.fit(X_train_seq.reshape(len(X_train_seq), -1), y_train_seq)
    pred_lgb = np.clip(
        lgb_model.predict(X_test_seq.reshape(len(X_test_seq), -1)), 0, 1
    )
    predictions["LightGBM"] = pred_lgb
    results["LightGBM"] = {
        "MSE": mean_squared_error(y_test_seq, pred_lgb),
        "MAE": mean_absolute_error(y_test_seq, pred_lgb),
    }

    base = persistence_scores(args.csv, timesteps=args.timesteps, horizon=args.horizon)
    results["Persistence"] = {"MSE": base["MSE"], "MAE": base["MAE"]}

    metrics_df = pd.DataFrame(results).T.sort_values("MSE")
    print("\n[Regression Metrics Comparison]")
    print(metrics_df)

    q1, q2 = np.quantile(y_train, 0.33), np.quantile(y_train, 0.66)
    y_true_cls = to_class(y_test_seq, q1, q2)
    # 기준선도 같은 표에 넣어야 비교가 된다.
    start = int(len(df) * 0.8) + args.timesteps
    predictions["Persistence"] = df["QoE_index"].values[start:]
    print(f"\n[Thresholds] Good(~{q1:.3f}) / Moderate(~{q2:.3f}) / Bad")
    print(f"\n{'Model':<15} | {'Accuracy':<8} | {'F1-Macro':<8}")
    print("-" * 40)
    for name, pred in predictions.items():
        pred_cls = to_class(pred, q1, q2)
        report = classification_report(y_true_cls, pred_cls, output_dict=True,
                                       zero_division=0)
        acc = accuracy_score(y_true_cls, pred_cls)
        print(f"{name:<15} | {acc:.4f}   | {report['macro avg']['f1-score']:.4f}")

    if args.plot:
        os.makedirs(args.outdir, exist_ok=True)

        plt.figure(figsize=(10, 5))
        metrics_df["MSE"].plot(kind="bar", color="skyblue", alpha=0.8)
        plt.title("Model MSE Comparison (lower is better)")
        plt.ylabel("MSE")
        plt.xticks(rotation=0)
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, "mse_comparison.png"), dpi=150)

        plt.figure(figsize=(15, 6))
        subset = 150
        plt.plot(y_test_seq[:subset], label="Actual", color="black",
                 linewidth=2, alpha=0.7)
        styles = {"GRU": ("red", "--"), "CNN-LSTM": ("blue", "-."),
                  "Transformer": ("green", ":"), "LightGBM": ("orange", "--"),
                  "Persistence": ("gray", "-")}
        for name, pred in predictions.items():
            color, style = styles.get(name, ("purple", "--"))
            plt.plot(pred[:subset], label=name, color=color, linestyle=style, alpha=0.8)
        plt.title(f"Future QoE prediction (first {subset} test samples)")
        plt.xlabel("Time step")
        plt.ylabel("QoE index (0~1)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, "prediction_comparison.png"), dpi=150)
        print(f"\n[INFO] 그래프 저장: {args.outdir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=DEFAULT_CSV)
    parser.add_argument("--timesteps", type=int, default=TIMESTEPS_DEFAULT)
    parser.add_argument("--horizon", type=int, default=1, help="몇 스텝(5분) 뒤를 예측할지")
    parser.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__), "figures"))
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    main(args)
