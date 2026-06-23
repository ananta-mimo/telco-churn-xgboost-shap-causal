"""
src/features/build_features.py
--------------------------------
Reusable feature engineering pipeline for the IBM Telco Churn project.
Called by:
  - 02_feature_engineering.ipynb  (fitting + transforming)
  - 03_modeling.ipynb              (transforming train/test splits)
  - app/app.py                     (transforming a single input row)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


# ── Constants ──────────────────────────────────────────────────────────────────
DROP_COLS = [
    "CustomerID", "Count", "Country", "State", "City",
    "Zip Code", "Lat Long", "Latitude", "Longitude",
    "Churn Score", "Churn Reason", "Churn Label",
]

SERVICE_COLS = [
    "Online Security", "Online Backup", "Device Protection",
    "Tech Support", "Streaming TV", "Streaming Movies",
]

BINARY_COLS = [
    "Partner", "Dependents", "Phone Service",
    "Paperless Billing", "Senior Citizen",
]

OHE_COLS = ["Internet Service", "Payment Method", "Gender"]

SCALE_COLS = ["Tenure Months", "Monthly Charges", "Total Charges",
              "Monthly Rate", "CLTV"]

TARGET = "Churn Value"


# ── Pipeline ───────────────────────────────────────────────────────────────────
def build_features(df: pd.DataFrame,
                   scaler: StandardScaler = None,
                   fit_scaler: bool = False,
                   return_target: bool = True):
    """
    Transform raw Telco dataframe into model-ready feature matrix.

    Parameters
    ----------
    df          : Raw dataframe as loaded from the Excel file.
    scaler      : Pre-fitted StandardScaler. Pass None to skip scaling.
    fit_scaler  : If True, fit the scaler on this data (training set only).
    return_target : If True, return (X, y). If False, return X only
                    (use for app inference where target is unknown).

    Returns
    -------
    X           : Feature DataFrame (unscaled).
    X_scaled    : Feature DataFrame (scaled numeric cols), or None if no scaler.
    y           : Target Series, or None if return_target=False.
    scaler      : Fitted scaler (same object passed in, or new one if fit_scaler=True).
    """
    df = df.copy()

    # 1. Drop non-predictive columns
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    df.drop(columns=cols_to_drop, inplace=True)

    # 2. Missing value treatment
    df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")
    df["Total Charges"].fillna(0.0, inplace=True)

    # 3. Derived: Monthly Rate
    df["Monthly Rate"] = np.where(
        df["Tenure Months"] > 0,
        df["Total Charges"] / df["Tenure Months"],
        df["Monthly Charges"],
    )

    # 4. Derived: Tenure Group (ordinal encoded)
    bins   = [0, 12, 24, 48, df["Tenure Months"].max() + 1]
    labels = ["0-12 mo", "13-24 mo", "25-48 mo", "49+ mo"]
    tenure_order = {"0-12 mo": 0, "13-24 mo": 1, "25-48 mo": 2, "49+ mo": 3}
    df["Tenure Group"] = pd.cut(df["Tenure Months"], bins=bins,
                                labels=labels, right=True)
    df["Tenure Group Enc"] = df["Tenure Group"].map(tenure_order)
    df.drop(columns=["Tenure Group"], inplace=True)

    # 5. Derived: Service Count
    for col in SERVICE_COLS:
        if col in df.columns:
            df[col + "_bin"] = (df[col] == "Yes").astype(int)
    bin_cols = [c + "_bin" for c in SERVICE_COLS if c + "_bin" in df.columns]
    df["Service Count"] = df[bin_cols].sum(axis=1)

    # 6. Derived: High Value Flag
    p75 = df["Monthly Charges"].quantile(0.75)
    df["High Value Flag"] = (df["Monthly Charges"] > p75).astype(int)

    # 7. Derived: Has Household
    df["Has Household"] = (
        (df.get("Partner", "No") == "Yes") |
        (df.get("Dependents", "No") == "Yes")
    ).astype(int)

    # 8. Binary encoding
    for col in BINARY_COLS:
        if col in df.columns:
            df[col + "_enc"] = (df[col] == "Yes").astype(int)

    if "Multiple Lines" in df.columns:
        df["Multiple Lines_enc"] = (df["Multiple Lines"] == "Yes").astype(int)

    # 9. Ordinal: Contract
    contract_order = {"Month-to-month": 0, "One year": 1, "Two year": 2}
    if "Contract" in df.columns:
        df["Contract_enc"] = df["Contract"].map(contract_order)

    # 10. One-hot encoding
    ohe_present = [c for c in OHE_COLS if c in df.columns]
    df = pd.get_dummies(df, columns=ohe_present, drop_first=True, dtype=int)

    # 11. Drop raw columns that have been encoded
    raw_cleanup = (
        BINARY_COLS
        + SERVICE_COLS
        + [c + "_bin" for c in SERVICE_COLS]
        + ["Partner", "Dependents", "Multiple Lines", "Contract"]
    )
    df.drop(columns=[c for c in raw_cleanup if c in df.columns], inplace=True)

    # 12. Separate target
    if return_target and TARGET in df.columns:
        y = df.pop(TARGET)
    else:
        y = None
        if TARGET in df.columns:
            df.drop(columns=[TARGET], inplace=True)

    X = df

    # 13. Scaling (numeric cols only)
    X_scaled = None
    if scaler is not None or fit_scaler:
        X_scaled = X.copy()
        scale_present = [c for c in SCALE_COLS if c in X_scaled.columns]
        if fit_scaler:
            scaler = StandardScaler()
            X_scaled[scale_present] = scaler.fit_transform(X[scale_present])
        else:
            X_scaled[scale_present] = scaler.transform(X[scale_present])

    return X, X_scaled, y, scaler
