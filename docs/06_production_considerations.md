# 06. Production Considerations
## IBM Telco Customer Churn

This document covers what happens after the notebooks: how to deploy the model,
keep it healthy over time, and make it useful to a business team beyond a local
Streamlit app.

---

## 1. When to Retrain the Model

A trained model is a snapshot of historical patterns. Customer behavior drifts,
competitors change offers, and the business introduces new products. The model
degrades silently unless monitored.

### Retrain triggers

| Trigger | Description | Suggested cadence |
|---------|-------------|-------------------|
| Scheduled retraining | Routine refresh regardless of drift signals | Monthly or quarterly |
| ROC-AUC drop | Model AUC on recent labeled data falls more than 0.03 below baseline | On detection |
| Churn rate shift | Observed churn rate in production deviates more than 5pp from training baseline | On detection |
| Feature distribution drift | PSI > 0.2 for any top-10 feature | Weekly check |
| Business event | New product launch, pricing change, competitor entry | Triggered manually |

### Population Stability Index (PSI)

PSI measures how much a feature's distribution has shifted between training and
current production data. It is the standard drift metric in industry.

```
PSI = sum((actual% - expected%) * ln(actual% / expected%))
```

| PSI value | Interpretation |
|-----------|---------------|
| PSI < 0.1 | No significant shift. Model stable. |
| 0.1 to 0.2 | Moderate shift. Monitor closely. |
| PSI > 0.2 | Significant shift. Retrain required. |

---


## 2. Model Versioning

Never overwrite a production model without versioning it first.

### Recommended naming convention

```
outputs/models/
├── xgb_model_v1_20240601.pkl       # original
├── xgb_model_v2_20240901.pkl       # after retraining
├── xgb_model_latest.pkl            # symlink or copy of current production model
├── feature_columns.json
└── model_metadata.json
```

### Metadata to log on every retrain

```json
{
  "model_version": "v2",
  "trained_date": "2024-09-01",
  "train_rows": 7043,
  "roc_auc_test": 0.847,
  "pr_auc": 0.683,
  "f1_at_threshold": 0.641,
  "optimal_threshold": 0.42,
  "data_date_range": "2024-01-01 to 2024-08-31",
  "top_drift_features": ["Monthly Charges", "Tenure Months"],
  "retrain_trigger": "scheduled_monthly"
}
```

---

## 3. Recommended Tests

Add a `tests/` folder with at minimum these three test files.

### tests/test_features.py

```python
import pandas as pd
from src.features.build_features import build_features

def test_output_shape():
    raw = pd.DataFrame([{
        "Tenure Months": 12, "Monthly Charges": 65.0,
        "Total Charges": 780.0, "Contract": "Month-to-month",
        "Internet Service": "Fiber optic", "Payment Method": "Electronic check",
        "Gender": "Male", "Senior Citizen": "No", "Partner": "No",
        "Dependents": "No", "Phone Service": "Yes", "Multiple Lines": "No",
        "Online Security": "No", "Online Backup": "No", "Device Protection": "No",
        "Tech Support": "No", "Streaming TV": "No", "Streaming Movies": "No",
        "Paperless Billing": "Yes", "Churn Value": 1, "CLTV": 3000,
    }])
    X, _, y, _ = build_features(raw)
    assert X.shape[0] == 1
    assert y is not None
    assert X.isnull().sum().sum() == 0

def test_no_nulls_after_engineering():
    raw = pd.DataFrame([{
        "Tenure Months": 0, "Monthly Charges": 29.0,
        "Total Charges": None, "Contract": "Two year",
        "Internet Service": "DSL", "Payment Method": "Mailed check",
        "Gender": "Female", "Senior Citizen": "No", "Partner": "Yes",
        "Dependents": "Yes", "Phone Service": "Yes", "Multiple Lines": "No",
        "Online Security": "Yes", "Online Backup": "Yes", "Device Protection": "Yes",
        "Tech Support": "Yes", "Streaming TV": "No", "Streaming Movies": "No",
        "Paperless Billing": "No", "Churn Value": 0, "CLTV": 5500,
    }])
    X, _, y, _ = build_features(raw)
    assert X.isnull().sum().sum() == 0
```

### tests/test_model.py

```python
import pickle, json
import pandas as pd

def test_model_loads():
    with open("outputs/models/xgb_model.pkl", "rb") as f:
        model = pickle.load(f)
    assert hasattr(model, "predict_proba")

def test_prediction_range():
    with open("outputs/models/xgb_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("outputs/models/feature_columns.json") as f:
        cols = json.load(f)
    X = pd.DataFrame([{c: 0 for c in cols}])
    prob = model.predict_proba(X)[0, 1]
    assert 0.0 <= prob <= 1.0
```

---

## 4. Limitations and Known Gaps

| Limitation | Impact | Mitigation |
|------------|--------|-----------|
| Static training data | Model degrades as behavior drifts | Monthly retraining + PSI monitoring |
| No real-time feature pipeline | App uses manual inputs only | Connect to CRM API for live customer data |
| Causal inference assumes no hidden confounders | DML ATE may be biased if unobserved variables drive both contract choice and churn | Sensitivity analysis using partial identification bounds |
| SMOTE on tabular data | Synthetic samples may not reflect real customer profiles | Evaluate with and without SMOTE, use class weights as alternative |
| Single model, no ensemble | XGBoost alone may be brittle on edge segments | Add a LightGBM or CatBoost challenger model |
| No feedback loop | Model never learns from retention campaign outcomes | Log campaign results and retrain with outcome labels |

---

## 5. Business Handoff Checklist

Before handing this model to a business team, confirm the following:

- [ ] Model metadata JSON is versioned and stored alongside the artifact
- [ ] Feature engineering is encapsulated in `build_features.py`, not duplicated
- [ ] The decision threshold is documented and justified, not left at 0.5
- [ ] SHAP explanations are validated against business intuition with a domain expert
- [ ] Causal findings are communicated as estimates with confidence intervals, not certainties
- [ ] A retraining schedule and drift monitoring process are agreed upon
- [ ] The data team has access to retrain without touching the app code
- [ ] Edge cases are documented: new customers with zero tenure, customers with no internet service
