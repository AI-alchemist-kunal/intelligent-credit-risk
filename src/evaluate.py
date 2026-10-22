
import numpy as np

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    roc_curve
)


def calculate_ks(y_true, probabilities):
    fpr, tpr, _ = roc_curve(
        y_true, probabilities
    )

    return float(np.max(tpr - fpr))


def evaluate_model(y_true, probabilities):
    return {
        "roc_auc": roc_auc_score(
            y_true, probabilities
        ),
        "average_precision": average_precision_score(
            y_true, probabilities
        ),
        "brier_score": brier_score_loss(
            y_true, probabilities
        ),
        "ks_statistic": calculate_ks(
            y_true, probabilities
        )
    }
