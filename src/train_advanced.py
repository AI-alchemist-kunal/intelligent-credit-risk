
import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.feature_engineering import create_features
from src.data_preprocessing import (
    build_preprocessor,
    FEATURE_COLUMNS
)

from src.train_baseline import evaluate


def main():

    df = pd.read_csv("data/raw/application_train.csv")
    df = create_features(df)

    X = df[FEATURE_COLUMNS]
    y = df["TARGET"].astype(int)

    X_dev, _, y_dev, _ = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_dev, y_dev,
        test_size=0.25,
        stratify=y_dev,
        random_state=42
    )

    models = {
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=42
        ),

        "LightGBM": LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=15,
            max_depth=-1,
            min_child_samples=40,
            verbosity=-1,
            n_jobs=-1,
            random_state=42
        )
    }

    results = {}

    for name, classifier in models.items():

        print(f"\nTraining {name}...")

        pipeline = Pipeline([
            ("preprocessing", build_preprocessor()),
            ("classifier", classifier)
        ])

        pipeline.fit(X_train, y_train)

        probabilities = pipeline.predict_proba(
            X_val
        )[:, 1]

        results[name] = evaluate(
            y_val,
            probabilities
        )

        filename = name.lower() + ".joblib"

        joblib.dump(
            pipeline,
            Path("models") / filename
        )

        print(name, results[name])

    Path("reports").mkdir(exist_ok=True)

    with open("reports/advanced_metrics.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
