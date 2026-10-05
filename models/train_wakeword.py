from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def load_dataset(csv_path: Path):
    df = pd.read_csv(csv_path)
    feature_cols = [c for c in df.columns if c.startswith("f_")]
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = (df["label"] == "positive").astype(int).to_numpy()
    return X, y, feature_cols


def train_model(
    csv_path: Path,
    model_output: Path,
    test_size: float = 0.2,
    random_state: int = 42,
) -> None:
    X, y, feature_cols = load_dataset(csv_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", probability=True)),
        ]
    )

    param_grid = {
        "clf__C": [0.1, 1.0, 10.0],
        "clf__gamma": ["scale", 0.01, 0.001],
    }

    search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=3,
        n_jobs=-1,
        scoring="f1",
        verbose=1,
    )
    search.fit(X_train, y_train)

    print("Best params:", search.best_params_)
    best_model = search.best_estimator_

    y_pred = best_model.predict(X_test)
    print("Classification report (0=negative, 1=positive):")
    print(classification_report(y_test, y_pred, digits=3))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Optimal eşik: precision >= 0.9 koşulu ile en yüksek recall veren threshold
    # (yanlış pozitifleri azaltmak için)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
    if len(thresholds) == 0:
        optimal_threshold = 0.80
    else:
        # precision >= 0.90 sağlayan threshold'lar arasından en yüksek recall
        candidates = [
            (t, p, r)
            for t, p, r in zip(thresholds, precisions[:-1], recalls[:-1])
            if p >= 0.90
        ]
        if candidates:
            best = max(candidates, key=lambda x: x[2])  # max recall
            optimal_threshold = float(np.clip(best[0], 0.55, 0.95))
        else:
            # precision 0.90’a ulaşılamıyorsa, en yüksek precision’lı threshold
            best_idx = np.argmax(precisions[:-1])
            optimal_threshold = float(np.clip(thresholds[best_idx], 0.60, 0.95))
    print(f"Optimal threshold (precision>=0.9): {optimal_threshold:.3f}")

    # n_mfcc: use_delta=True → features = n_mfcc × 6
    n_mfcc = len(feature_cols) // 6

    model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": best_model,
            "feature_cols": feature_cols,
            "prob_threshold": optimal_threshold,
            "n_mfcc": n_mfcc,
        },
        model_output,
    )
    print(f"Saved model to {model_output} (threshold={optimal_threshold:.3f}, n_mfcc={n_mfcc})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SVM wake-word model on MFCC CSV.")
    parser.add_argument(
        "--csv",
        type=str,
        default="data/csv/mfcc_features.csv",
        help="Input MFCC CSV file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/wakeword_model.joblib",
        help="Output model path.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test split fraction.",
    )
    args = parser.parse_args()

    train_model(Path(args.csv), Path(args.output), test_size=args.test_size)


if __name__ == "__main__":
    main()

