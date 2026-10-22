
from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss
)

from src.feature_engineering import create_features
from src.data_preprocessing import (
    build_preprocessor,
    FEATURE_COLUMNS
)


def evaluate(y_true, probabilities):
    return {
        "roc_auc": float(
            roc_auc_score(y_true, probabilities)
        ),
        "average_precision": float(
            average_precision_score(y_true, probabilities)
        ),
        "brier_score": float(
            brier_score_loss(y_true, probabilities)
        ),
        "log_loss": float(
            log_loss(y_true, probabilities)
        )
    }


def main():

    data_path = Path("data/raw/application_train.csv")

    df = pd.read_csv(data_path)
    df = create_features(df)

    X = df[FEATURE_COLUMNS]
    y = df["TARGET"].astype(int)

    # Separate an untouched test set
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    # 25% of remaining 80% = 20% of total
    X_train, X_val, y_train, y_val = train_test_split(
        X_dev, y_dev,
        test_size=0.25,
        stratify=y_dev,
        random_state=42
    )

    model = Pipeline([
        ("preprocessing", build_preprocessor()),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                solver="liblinear",
                random_state=42
            )
        )
    ])

    print("Training Logistic Regression...")
    model.fit(X_train, y_train)

    # Evaluate only on validation for now
    val_prob = model.predict_proba(X_val)[:, 1]

    results = {
        "model": "Logistic Regression",
        "validation": evaluate(y_val, val_prob),
        "validation_target_rate": float(y_val.mean()),
        "data_split": "stratified random 60/20/20"
    }

    print(json.dumps(results, indent=2))

    Path("models").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)

    joblib.dump(model, "models/logistic_baseline.joblib")

    with open("reports/baseline_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Baseline training complete.")


if __name__ == "__main__":
    main()
