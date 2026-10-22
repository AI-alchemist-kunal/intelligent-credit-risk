
import numpy as np
import pandas as pd

from src.feature_engineering import create_features


def test_feature_engineering():

    sample = pd.DataFrame({
        "AMT_INCOME_TOTAL": [100000, 0],
        "AMT_CREDIT": [50000, 10000],
        "AMT_ANNUITY": [5000, 1000],
        "DAYS_BIRTH": [-12000, -14000],
        "DAYS_EMPLOYED": [-1000, 365243]
    })

    result = create_features(sample)

    assert result.loc[0, "CREDIT_INCOME_RATIO"] == 0.5

    assert np.isnan(
        result.loc[1, "EMPLOYMENT_YEARS"]
    )

    assert result["CREDIT_INCOME_RATIO"].notna().sum() == 1
