
import numpy as np
import pandas as pd


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Home Credit employment sentinel
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(
        365243, np.nan
    )

    # Income denominators must be nonzero
    income = df["AMT_INCOME_TOTAL"].replace(0, np.nan)

    # Requested credit relative to annual income
    df["CREDIT_INCOME_RATIO"] = (
        df["AMT_CREDIT"] / income
    )

    # Annualized installment relative to annual income
    df["ANNUITY_INCOME_RATIO"] = (
        12 * df["AMT_ANNUITY"] / income
    )

    # Annuity relative to loan principal
    df["ANNUITY_CREDIT_RATIO"] = (
        df["AMT_ANNUITY"]
        / df["AMT_CREDIT"].replace(0, np.nan)
    )

    # Home Credit records birth age in negative days
    df["AGE_YEARS"] = (
        -df["DAYS_BIRTH"] / 365.25
    )

    # Employment duration, excluding the anomalous value
    df["EMPLOYMENT_YEARS"] = (
        -df["DAYS_EMPLOYED"] / 365.25
    )

    # Replace invalid numeric calculation results
    df = df.replace([np.inf, -np.inf], np.nan)

    return df
