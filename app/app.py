"""
app/app.py
-----------
Streamlit dashboard for the IBM Telco Customer Churn project.

Run from the project root:
    streamlit run app/app.py

Pages:
  1. Overview       - project summary + key EDA charts
  2. Churn Predictor - individual customer scoring + SHAP waterfall
  3. Model Performance - evaluation metrics + curves
"""

import os, json, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import streamlit as st

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "outputs", "models")
FIG_DIR   = os.path.join(ROOT, "outputs", "figures")
DATA_PATH = os.path.join(ROOT, "data",    "Telco_customer_churn.xlsx")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "Telco Churn Intelligence",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  .metric-card {
    background: #0f1117;
    border: 1px solid #2d2d2d;
    border-radius: 10px;
    padding: 20px 24px;
    text-align: center;
  }
  .metric-card .label {
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #888;
    margin-bottom: 6px;
  }
  .metric-card .value {
    font-size: 2rem;
    font-weight: 700;
    color: #ffffff;
    font-family: 'JetBrains Mono', monospace;
  }
  .metric-card .sub {
    font-size: 0.78rem;
    color: #666;
    margin-top: 4px;
  }

  .risk-high {
    background: linear-gradient(135deg, #3a0a0a, #1a0505);
    border: 1px solid #c0392b;
    border-radius: 12px;
    padding: 24px;
    text-align: center;
  }
  .risk-low {
    background: linear-gradient(135deg, #0a2a0a, #051505);
    border: 1px solid #27ae60;
    border-radius: 12px;
    padding: 24px;
    text-align: center;
  }
  .risk-mid {
    background: linear-gradient(135deg, #2a2000, #151000);
    border: 1px solid #f39c12;
    border-radius: 12px;
    padding: 24px;
    text-align: center;
  }
  .risk-label {
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 8px;
  }
  .risk-prob {
    font-size: 3.2rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    line-height: 1;
  }
  .risk-desc { font-size: 0.82rem; color: #aaa; margin-top: 10px; }

    .section-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #1e293b;        ← dark slate, readable on both light and dark
        border-left: 3px solid #4878CF;
        padding-left: 10px;
        margin: 20px 0 12px 0;
    }

  div[data-testid="stSidebar"] {
    background: #0a0a0f;
    border-right: 1px solid #1e1e2e;
  }
</style>
""", unsafe_allow_html=True)


# ── Loaders ───────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open(f"{MODEL_DIR}/xgb_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(f"{MODEL_DIR}/feature_columns.json") as f:
        feature_cols = json.load(f)
    with open(f"{MODEL_DIR}/model_metadata.json") as f:
        metadata = json.load(f)
    return model, feature_cols, metadata


@st.cache_data
def load_data():
    df = pd.read_excel(DATA_PATH)
    return df


def fig_path(name):
    return os.path.join(FIG_DIR, name)


# ── Feature engineering (mirrors notebook 02) ─────────────────────────────────
def engineer_features(raw: dict, feature_cols: list) -> pd.DataFrame:
    """
    Transform raw customer inputs into model-ready feature vector.
    Mirrors the logic in src/features/build_features.py.
    """
    d = raw.copy()

    # Derived
    tenure = float(d.get("Tenure Months", 1))
    total  = float(d.get("Total Charges", 0))
    monthly = float(d.get("Monthly Charges", 50))

    d["Monthly Rate"]     = total / tenure if tenure > 0 else monthly
    d["High Value Flag"]  = 1 if monthly > 71.0 else 0   # approximate p75

    # Tenure Group ordinal
    if   tenure <= 12: d["Tenure Group Enc"] = 0
    elif tenure <= 24: d["Tenure Group Enc"] = 1
    elif tenure <= 48: d["Tenure Group Enc"] = 2
    else:              d["Tenure Group Enc"] = 3

    # Service Count
    service_cols = ["Online Security", "Online Backup", "Device Protection",
                    "Tech Support", "Streaming TV", "Streaming Movies"]
    d["Service Count"] = sum(1 for c in service_cols if d.get(c) == "Yes")

    # Has Household
    d["Has Household"] = 1 if (d.get("Partner") == "Yes" or
                                d.get("Dependents") == "Yes") else 0

    # Binary encodings
    binary_map = {
        "Partner_enc"          : d.get("Partner")          == "Yes",
        "Dependents_enc"       : d.get("Dependents")       == "Yes",
        "Phone Service_enc"    : d.get("Phone Service")    == "Yes",
        "Paperless Billing_enc": d.get("Paperless Billing") == "Yes",
        "Senior Citizen_enc"   : d.get("Senior Citizen")   == "Yes",
        "Multiple Lines_enc"   : d.get("Multiple Lines")   == "Yes",
    }
    for k, v in binary_map.items():
        d[k] = int(v)

    # Contract ordinal
    contract_order = {"Month-to-month": 0, "One year": 1, "Two year": 2}
    d["Contract_enc"] = contract_order.get(d.get("Contract", "Month-to-month"), 0)

    # One-hot: Internet Service
    for val in ["Fiber optic", "No"]:
        col = f"Internet Service_{val}"
        d[col] = 1 if d.get("Internet Service") == val else 0

    # One-hot: Payment Method
    for val in ["Credit card (automatic)", "Electronic check", "Mailed check"]:
        col = f"Payment Method_{val}"
        d[col] = 1 if d.get("Payment Method") == val else 0

    # One-hot: Gender
    d["Gender_Male"] = 1 if d.get("Gender") == "Male" else 0

    # Build row aligned to feature_cols
    row = {col: d.get(col, 0) for col in feature_cols}
    return pd.DataFrame([row])[feature_cols].astype(float)


# ── Gauge chart ───────────────────────────────────────────────────────────────
def draw_gauge(prob: float) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4, 2.2), subplot_kw={"polar": True})
    fig.patch.set_facecolor("#0f1117")

    theta_start, theta_end = np.pi, 0
    theta = theta_start + (theta_end - theta_start) * prob

    # Background arc
    arc_theta = np.linspace(np.pi, 0, 200)
    ax.plot(arc_theta, [1]*200, color="#2d2d2d", lw=18, solid_capstyle="round",
            transform=ax.transData)

    # Color zones
    for start, end, color in [
        (np.pi,       np.pi*2/3, "#27ae60"),
        (np.pi*2/3,   np.pi/3,   "#f39c12"),
        (np.pi/3,     0,         "#c0392b"),
    ]:
        t = np.linspace(start, end, 100)
        ax.plot(t, [1]*100, color=color, lw=18, solid_capstyle="butt",
                transform=ax.transData)

    # Needle
    ax.annotate("", xy=(theta, 0.85), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->, head_width=0.15, head_length=0.1",
                                color="white", lw=2))

    # Label
    ax.text(0, -0.3, f"{prob*100:.1f}%", ha="center", va="center",
            fontsize=18, fontweight="bold", color="white",
            fontfamily="monospace", transform=ax.transData)

    ax.set_ylim(0, 1.3)
    ax.set_facecolor("#0f1117")
    ax.axis("off")
    plt.tight_layout(pad=0)
    return fig


# ── Page 1: Overview ──────────────────────────────────────────────────────────
def page_overview():
    st.markdown("## Project Overview")
    st.markdown(
        "IBM Telco Customer Churn analysis across **7,043 customers** and **33 features**. "
        "Combines predictive modeling, SHAP explainability, and causal inference to surface "
        "actionable retention insights."
    )

    # KPI row
    try:
        df = load_data()
        churn_rate = df["Churn Value"].mean() * 100
        avg_cltv   = df["CLTV"].mean()
        avg_tenure = df["Tenure Months"].mean()
        mtm_rate   = (df["Contract"] == "Month-to-month").mean() * 100
    except Exception:
        churn_rate, avg_cltv, avg_tenure, mtm_rate = 26.5, 3978, 32.4, 55.0

    c1, c2, c3, c4 = st.columns(4)
    for col, label, value, sub in [
        (c1, "OVERALL CHURN RATE",    f"{churn_rate:.1f}%",  "Moderately imbalanced target"),
        (c2, "AVG CUSTOMER CLTV",     f"${avg_cltv:,.0f}",   "Predicted lifetime value"),
        (c3, "AVG TENURE",            f"{avg_tenure:.0f} mo", "Months as a customer"),
        (c4, "MONTH-TO-MONTH SHARE",  f"{mtm_rate:.0f}%",    "Highest churn risk tier"),
    ]:
        col.markdown(
            f'<div class="metric-card">'
            f'<div class="label">{label}</div>'
            f'<div class="value">{value}</div>'
            f'<div class="sub">{sub}</div>'
            f'</div>', unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts row 1
    st.markdown('<div class="section-header">Churn by Key Segments</div>',
                unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    for col, fname, caption in [
        (col1, "02_churn_by_contract.png",  "Contract Type"),
        (col2, "03_churn_by_tenure.png",    "Tenure Group"),
        (col3, "04_churn_by_payment.png",   "Payment Method"),
    ]:
        p = fig_path(fname)
        if os.path.exists(p):
            col.image(p, caption=caption, use_container_width=True)
        else:
            col.info(f"Run notebook 01 to generate: {fname}")

    # Charts row 2
    st.markdown('<div class="section-header">Service and Value Analysis</div>',
                unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    for col, fname, caption in [
        (col1, "05_churn_by_internet.png",       "Internet Service"),
        (col2, "06_churn_heatmap_internet_contract.png", "Contract x Internet Heatmap"),
        (col3, "09_cltv_churn_score.png",        "CLTV vs Churn Score"),
    ]:
        p = fig_path(fname)
        if os.path.exists(p):
            col.image(p, caption=caption, use_container_width=True)
        else:
            col.info(f"Run notebook 01 to generate: {fname}")

    # Key findings
    st.markdown('<div class="section-header">Key Findings</div>', unsafe_allow_html=True)
    findings = [
        (" ", "Month-to-month customers churn at 42%", "vs 3% for two-year contracts. Contract type is the dominant churn driver."),
        (" ", "First 12 months are the danger zone", "New customers churn at ~50%. Early-tenure engagement has the highest ROI."),
        (" ", "Electronic check users churn at 45%", "Over 2x the rate of auto-pay customers. Auto-pay nudges are a low-cost lever."),
        (" ", "Fiber optic customers churn at 42%", "Premium product, premium churn. Review competitive pricing."),
        (" ", "Churned customers have higher median CLTV", "The business is losing its most valuable customers at above-average rates."),
        (" ", "Competitor offers dominate churn reasons", "Retention strategy must include competitive counter-offers, not just service fixes."),
    ]
 
    cols = st.columns(3)
    for i, (icon, title, desc) in enumerate(findings):
        with cols[i % 3]:
            st.markdown(f"**{icon} {title}**")
            st.caption(desc)
            st.markdown("")


# ── Page 2: Predictor ─────────────────────────────────────────────────────────
def page_predictor():
    st.markdown("## Churn Risk Predictor")
    st.markdown(
        "Enter a customer's details to generate a real-time churn probability "
        "and a SHAP explanation of which factors drive the score."
    )

    try:
        model, feature_cols, metadata = load_model()
        threshold = metadata.get("optimal_threshold", 0.5)
    except FileNotFoundError:
        st.error("Model not found. Run notebook 03 first to train and save the model.")
        return

    # ── Input form ────────────────────────────────────────────────────────────
    with st.form("predictor_form"):
        st.markdown('<div class="section-header">Account Details</div>',
                    unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        tenure         = c1.slider("Tenure (months)", 0, 72, 12)
        monthly_charges= c2.slider("Monthly Charges ($)", 18.0, 120.0, 65.0, step=0.5)
        total_charges  = c3.number_input("Total Charges ($)", 0.0, 10000.0,
                                          float(tenure * monthly_charges), step=10.0)

        st.markdown('<div class="section-header">Contract and Billing</div>',
                    unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        contract       = c1.selectbox("Contract",
                                       ["Month-to-month", "One year", "Two year"])
        payment_method = c2.selectbox("Payment Method",
                                       ["Electronic check", "Mailed check",
                                        "Bank transfer (automatic)",
                                        "Credit card (automatic)"])
        paperless      = c3.checkbox("Paperless Billing", value=True)

        st.markdown('<div class="section-header">Services</div>',
                    unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        internet       = c1.selectbox("Internet Service",
                                       ["Fiber optic", "DSL", "No"])
        phone          = c2.checkbox("Phone Service", value=True)
        multi_lines    = c3.checkbox("Multiple Lines", value=False)
        online_sec     = c4.checkbox("Online Security", value=False)

        c1, c2, c3, c4 = st.columns(4)
        online_back    = c1.checkbox("Online Backup", value=False)
        device_prot    = c2.checkbox("Device Protection", value=False)
        tech_support   = c3.checkbox("Tech Support", value=False)
        stream_tv      = c4.checkbox("Streaming TV", value=False)
        stream_movies  = st.checkbox("Streaming Movies", value=False)

        st.markdown('<div class="section-header">Demographics</div>',
                    unsafe_allow_html=True)
        c1, c2, c3, c4, c5 = st.columns(5)
        gender         = c1.selectbox("Gender", ["Male", "Female"])
        senior         = c2.checkbox("Senior Citizen", value=False)
        partner        = c3.checkbox("Partner", value=False)
        dependents     = c4.checkbox("Dependents", value=False)
        cltv           = c5.number_input("CLTV ($)", 0.0, 8000.0, 3000.0, step=50.0)

        submitted = st.form_submit_button("Calculate Churn Risk", type="primary",
                                          use_container_width=True)

    if not submitted:
        st.info("Fill in customer details above and click **Calculate Churn Risk**.")
        return

    # ── Build input row ───────────────────────────────────────────────────────
    raw = {
        "Tenure Months"     : tenure,
        "Monthly Charges"   : monthly_charges,
        "Total Charges"     : total_charges,
        "CLTV"              : cltv,
        "Contract"          : contract,
        "Payment Method"    : payment_method,
        "Paperless Billing" : "Yes" if paperless    else "No",
        "Internet Service"  : internet,
        "Phone Service"     : "Yes" if phone        else "No",
        "Multiple Lines"    : "Yes" if multi_lines  else "No",
        "Online Security"   : "Yes" if online_sec   else "No",
        "Online Backup"     : "Yes" if online_back  else "No",
        "Device Protection" : "Yes" if device_prot  else "No",
        "Tech Support"      : "Yes" if tech_support else "No",
        "Streaming TV"      : "Yes" if stream_tv    else "No",
        "Streaming Movies"  : "Yes" if stream_movies else "No",
        "Gender"            : gender,
        "Senior Citizen"    : "Yes" if senior       else "No",
        "Partner"           : "Yes" if partner      else "No",
        "Dependents"        : "Yes" if dependents   else "No",
    }

    X_input = engineer_features(raw, feature_cols)
    prob    = float(model.predict_proba(X_input)[0, 1])
    label   = "HIGH RISK" if prob >= threshold else ("MEDIUM RISK" if prob >= 0.3 else "LOW RISK")
    css_cls = "risk-high" if prob >= threshold else ("risk-mid" if prob >= 0.3 else "risk-low")
    color   = "#c0392b" if prob >= threshold else ("#f39c12" if prob >= 0.3 else "#27ae60")

    desc_map = {
        "HIGH RISK"  : f"Above the {threshold:.0%} decision threshold. Prioritise for retention outreach.",
        "MEDIUM RISK": "Monitor closely. Consider a proactive check-in or offer.",
        "LOW RISK"   : "Below the action threshold. Standard engagement is sufficient.",
    }

    # ── Result display ────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    col_gauge, col_card, col_summary = st.columns([1.2, 1, 1.8])

    with col_gauge:
        st.markdown("**Churn Probability Gauge**")
        gauge_fig = draw_gauge(prob)
        st.pyplot(gauge_fig, use_container_width=True)
        plt.close(gauge_fig)

    with col_card:
        st.markdown(
            f'<div class="{css_cls}">'
            f'<div class="risk-label" style="color:{color}">{label}</div>'
            f'<div class="risk-prob" style="color:{color}">{prob*100:.1f}%</div>'
            f'<div class="risk-desc">{desc_map[label]}</div>'
            f'</div>', unsafe_allow_html=True
        )

    with col_summary:
        st.markdown("**Customer Profile Summary**")
        summary = {
            "Contract"       : contract,
            "Tenure"         : f"{tenure} months",
            "Monthly Charges": f"${monthly_charges:.2f}",
            "Internet"       : internet,
            "Service Count"  : int(sum([online_sec, online_back, device_prot,
                                         tech_support, stream_tv, stream_movies])),
            "Payment"        : payment_method,
            "Household"      : "Yes" if (partner or dependents) else "No",
            "Senior"         : "Yes" if senior else "No",
        }
        for k, v in summary.items():
            st.markdown(f"**{k}:** {v}")

    # ── SHAP explanation ──────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Why This Score: SHAP Explanation</div>',
                unsafe_allow_html=True)
    st.caption(
        "Each bar shows how much a feature pushed the score above (red) or "
        "below (blue) the model baseline. The sum of all bars equals the final prediction."
    )

    try:
        booster  = model.get_booster()
        booster.set_param({"base_score": 0.5})
        explainer   = shap.TreeExplainer(booster)
        shap_values = explainer.shap_values(X_input)

        shap_series = pd.Series(shap_values[0], index=feature_cols)
        top_shap    = shap_series.abs().sort_values(ascending=False).head(12)
        shap_top    = shap_series[top_shap.index].sort_values()

        fig, ax = plt.subplots(figsize=(9, 5))
        fig.patch.set_facecolor("#0f1117")
        ax.set_facecolor("#0f1117")

        colors = ["#c0392b" if v > 0 else "#4878CF" for v in shap_top.values]
        bars   = ax.barh(shap_top.index, shap_top.values,
                         color=colors, edgecolor="none", height=0.65)
        ax.axvline(0, color="#888", lw=0.8)
        ax.set_xlabel("SHAP Value (impact on churn probability)", color="#ccc")
        ax.tick_params(colors="#ccc")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for spine in ["bottom", "left"]:
            ax.spines[spine].set_color("#444")

        for bar, val in zip(bars, shap_top.values):
            offset = 0.002 if val >= 0 else -0.002
            ha     = "left" if val >= 0 else "right"
            ax.text(val + offset, bar.get_y() + bar.get_height()/2,
                    f"{val:+.3f}", va="center", ha=ha, fontsize=8, color="#ddd")

        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    except Exception as e:
        st.warning(f"SHAP explanation unavailable: {e}")


# ── Page 3: Model Performance ─────────────────────────────────────────────────
def page_model():
    st.markdown("## Model Performance")

    try:
        _, _, metadata = load_model()
    except FileNotFoundError:
        st.error("Model not found. Run notebook 03 first.")
        return

    # Metrics
    st.markdown('<div class="section-header">Evaluation Metrics</div>',
                unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    metrics = [
        (c1, "ROC-AUC",      f"{metadata.get('roc_auc_test', 0):.4f}",    "Test set"),
        (c2, "ROC-AUC (CV)", f"{metadata.get('roc_auc_cv',   0):.4f}",    "5-fold CV"),
        (c3, "PR-AUC",       f"{metadata.get('pr_auc',       0):.4f}",    "Imbalanced target"),
        (c4, "F1 Score",     f"{metadata.get('f1_at_best_threshold', 0):.4f}", "At tuned threshold"),
        (c5, "Threshold",    f"{metadata.get('optimal_threshold', 0.5):.2f}", "Decision cutoff"),
    ]
    for col, label, value, sub in metrics:
        col.markdown(
            f'<div class="metric-card">'
            f'<div class="label">{label}</div>'
            f'<div class="value">{value}</div>'
            f'<div class="sub">{sub}</div>'
            f'</div>', unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts
    st.markdown('<div class="section-header">ROC and Precision-Recall Curves</div>',
                unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    roc_path = fig_path("14_roc_pr_curves.png")
    cm_path  = fig_path("15_confusion_matrices.png")
    if os.path.exists(roc_path):
        c1.image(roc_path, caption="ROC and PR Curves", use_container_width=True)
    else:
        c1.info("Run notebook 03 to generate curves.")
    if os.path.exists(cm_path):
        c2.image(cm_path, caption="Confusion Matrices", use_container_width=True)
    else:
        c2.info("Run notebook 03 to generate confusion matrices.")

    st.markdown('<div class="section-header">Threshold Tuning and Feature Importance</div>',
                unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    thresh_path = fig_path("16_threshold_tuning.png")
    imp_path    = fig_path("17_feature_importance.png")
    if os.path.exists(thresh_path):
        c1.image(thresh_path, caption="Threshold vs F1/Precision/Recall", use_container_width=True)
    else:
        c1.info("Run notebook 03 to generate threshold plot.")
    if os.path.exists(imp_path):
        c2.image(imp_path, caption="XGBoost Feature Importance (Gain)", use_container_width=True)
    else:
        c2.info("Run notebook 03 to generate importance plot.")

    st.markdown('<div class="section-header">SHAP Global Explanation</div>',
                unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    bee_path = fig_path("18_shap_beeswarm.png")
    bar_path = fig_path("19_shap_bar.png")
    if os.path.exists(bee_path):
        c1.image(bee_path, caption="SHAP Beeswarm (Global)", use_container_width=True)
    if os.path.exists(bar_path):
        c2.image(bar_path, caption="Mean |SHAP| Values", use_container_width=True)

    st.markdown('<div class="section-header">Model Configuration</div>',
                unsafe_allow_html=True)
    config = {
        "Algorithm"         : "XGBoost",
        "Training rows"     : f"{metadata.get('train_rows', 0):,} (after SMOTE)",
        "Test rows"         : f"{metadata.get('test_rows', 0):,}",
        "Number of features": metadata.get("n_features", 0),
        "Optimal threshold" : metadata.get("optimal_threshold", 0.5),
        "Imbalance handling": "SMOTE on training data only",
        "Baseline model"    : "Logistic Regression (for comparison)",
    }
    for k, v in config.items():
        st.markdown(f"**{k}:** {v}")


# ── Sidebar navigation ────────────────────────────────────────────────────────
def main():
    with st.sidebar:
        st.markdown("### Telco Churn")
        st.markdown("**Intelligence Dashboard**")
        st.markdown("---")

        page = st.radio(
            "Navigate",
            ["Overview", "Churn Predictor", "Model Performance"],
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.caption("IBM Telco Customer Churn")
        st.caption("XGBoost + SHAP + DML")
        st.caption("7,043 customers, 33 features")

    if   page == "Overview":         page_overview()
    elif page == "Churn Predictor":  page_predictor()
    elif page == "Model Performance": page_model()


if __name__ == "__main__":
    main()
