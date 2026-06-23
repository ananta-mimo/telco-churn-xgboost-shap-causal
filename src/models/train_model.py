"""
src/models/train_model.py
--------------------------
Reusable training function for the IBM Telco Churn XGBoost model.
Called by 03_modeling.ipynb and any retraining scripts.
"""

import pandas as pd
import numpy as np
import pickle, json, os
from sklearn.model_selection import train_test_split
from sklearn.metrics         import roc_auc_score, average_precision_score, f1_score
from imblearn.over_sampling  import SMOTE
from xgboost                 import XGBClassifier


SEED      = 42
MODEL_DIR = "../outputs/models"


def train(X: pd.DataFrame, y: pd.Series,
          test_size: float = 0.20,
          n_estimators: int = 400,
          max_depth: int = 4,
          learning_rate: float = 0.05) -> dict:
    """
    Train XGBoost on SMOTE-balanced data and return model + metrics.

    Returns
    -------
    dict with keys: model, X_test, y_test, y_prob, metrics
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=SEED
    )

    smote = SMOTE(random_state=SEED)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

    neg = (y_train_sm == 0).sum()
    pos = (y_train_sm == 1).sum()

    model = XGBClassifier(
        n_estimators     = n_estimators,
        max_depth        = max_depth,
        learning_rate    = learning_rate,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        min_child_weight = 5,
        scale_pos_weight = neg / pos,
        eval_metric      = "auc",
        use_label_encoder= False,
        random_state     = SEED,
        n_jobs           = -1,
    )
    model.fit(X_train_sm, y_train_sm,
              eval_set=[(X_test, y_test)],
              verbose=False)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "roc_auc" : round(roc_auc_score(y_test, y_prob), 4),
        "pr_auc"  : round(average_precision_score(y_test, y_prob), 4),
        "f1"      : round(f1_score(y_test, y_pred), 4),
    }

    return {
        "model"  : model,
        "X_test" : X_test,
        "y_test" : y_test,
        "y_prob" : y_prob,
        "metrics": metrics,
    }


def save_model(model: XGBClassifier,
               feature_columns: list,
               metadata: dict,
               model_dir: str = MODEL_DIR) -> None:
    """Persist model, feature list, and metadata to disk."""
    os.makedirs(model_dir, exist_ok=True)

    with open(f"{model_dir}/xgb_model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open(f"{model_dir}/feature_columns.json", "w") as f:
        json.dump(feature_columns, f, indent=2)

    with open(f"{model_dir}/model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Model artifacts saved to {model_dir}/")


def load_model(model_dir: str = MODEL_DIR) -> tuple:
    """Load model, feature columns, and metadata from disk."""
    with open(f"{model_dir}/xgb_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open(f"{model_dir}/feature_columns.json") as f:
        feature_columns = json.load(f)

    with open(f"{model_dir}/model_metadata.json") as f:
        metadata = json.load(f)

    return model, feature_columns, metadata
