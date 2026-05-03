"""Streamlit dashboard for credit risk prediction.

Run with: streamlit run app/dashboard.py
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from src.drift import compute_psi, simulate_drift

st.set_page_config(page_title="Credit Scoring Dashboard", layout="wide")

# --- Load artifacts ---
@st.cache_resource
def load_models():
    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    reports_dir = os.path.join(os.path.dirname(__file__), "..", "reports")
    deploy = joblib.load(os.path.join(models_dir, "deploy_config.pkl"))
    ebm = joblib.load(os.path.join(models_dir, "ebm_top10.pkl"))
    xgb = joblib.load(os.path.join(models_dir, "xgb_top10.pkl"))
    explainer = joblib.load(os.path.join(models_dir, "shap_explainer.pkl"))
    fairness = pd.read_csv(os.path.join(reports_dir, "fairness_summary.csv"), index_col=0)
    # Drift data (may not exist if notebook 06 hasn't been run)
    drift_temporal = None
    drift_perf = None
    drift_samples = None
    drift_path = os.path.join(reports_dir, "drift_temporal.csv")
    if os.path.exists(drift_path):
        drift_temporal = pd.read_csv(drift_path)
        drift_perf = pd.read_csv(os.path.join(reports_dir, "drift_performance.csv"))
        drift_samples = pd.read_csv(os.path.join(reports_dir, "drift_samples.csv"))
    return deploy, ebm, xgb, explainer, fairness, drift_temporal, drift_perf, drift_samples

deploy, ebm, xgb, explainer, fairness_summary, drift_temporal, drift_perf, drift_samples = load_models()

# --- Sidebar: Borrower Inputs ---
st.sidebar.header("Borrower Information")
descs = deploy["feature_descriptions"]

inputs = {
    "int_rate": st.sidebar.slider(descs["int_rate"], 5.0, 30.0, 12.0, 0.1),
    "term": st.sidebar.selectbox(descs["term"], [36, 60], index=0),
    "acc_open_past_24mths": st.sidebar.slider(descs["acc_open_past_24mths"], 0, 20, 3),
    "annual_inc": st.sidebar.number_input(descs["annual_inc"], 10000, 500000, 60000, 5000),
    "dti": st.sidebar.slider(descs["dti"], 0.0, 50.0, 15.0, 0.5),
    "fico": st.sidebar.slider(descs["fico"], 300, 850, 700),
    "loan_amnt": st.sidebar.number_input(descs["loan_amnt"], 1000, 40000, 10000, 500),
    "tot_hi_cred_lim": st.sidebar.number_input(descs["tot_hi_cred_lim"], 0, 1000000, 100000, 5000),
    "avg_cur_bal": st.sidebar.number_input(descs["avg_cur_bal"], 0, 500000, 10000, 1000),
    "total_bc_limit": st.sidebar.number_input(descs["total_bc_limit"], 0, 200000, 20000, 1000),
}

X_input = pd.DataFrame([inputs])
# Streamlit number_input returns int when all bounds are int, but the saved
# Pipeline's SimpleImputer was fitted with float fill values (NaN forces float
# dtype at fit time). Cast all integer columns to float64 to match.
for _col in X_input.select_dtypes(include="integer").columns:
    X_input[_col] = X_input[_col].astype("float64")

# --- Main: Prediction ---
st.title("Credit Scoring Dashboard")

tab1, tab2, tab3, tab4 = st.tabs(["Prediction", "Explanation", "Fairness", "Monitoring"])

with tab1:
    st.header("Loan Decision")

    prob = ebm.predict_proba(X_input)[:, 1][0]
    threshold = deploy["threshold"]
    decision = "DENIED" if prob >= threshold else "APPROVED"
    color = "red" if decision == "DENIED" else "green"

    col1, col2, col3 = st.columns(3)
    col1.metric("Default Probability", f"{prob:.1%}")
    col2.metric("Threshold", f"{threshold:.1%}")
    col3.markdown(f"### Decision: :{color}[{decision}]")

    # Probability gauge
    fig, ax = plt.subplots(figsize=(8, 1.5))
    ax.barh(0, prob, color="coral", height=0.5)
    ax.barh(0, 1 - prob, left=prob, color="steelblue", height=0.5)
    ax.axvline(x=threshold, color="black", linestyle="--", linewidth=2, label=f"Threshold ({threshold:.1%})")
    ax.set_xlim(0, 1); ax.set_yticks([]); ax.legend(loc="upper right")
    ax.set_xlabel("P(Default)")
    st.pyplot(fig)
    plt.close()

with tab2:
    st.header("SHAP Explanation")
    st.write("Feature contributions to the prediction (using XGBoost SHAP explainer):")

    # The TreeExplainer was built on xgb.named_steps["clf"], which expects
    # the post-preprocessor feature matrix (10 raw columns -> ~12 after OHE
    # of `term`). Transform through the Pipeline's preprocessor first, then
    # aggregate OHE-expanded SHAP contributions back to the raw feature names
    # so the waterfall plot stays human-readable.
    import numpy as _np
    pre = xgb.named_steps["pre"]
    X_pre = pre.transform(X_input)
    raw_feat_names = deploy["features"]
    pre_feat_names = list(pre.get_feature_names_out())

    shap_vals_pre = explainer.shap_values(X_pre)

    # Aggregate post-OHE columns back to raw features by name match.
    agg = _np.zeros(len(raw_feat_names), dtype=float)
    for i, raw in enumerate(raw_feat_names):
        for j, post in enumerate(pre_feat_names):
            stripped = post.split("__", 1)[-1]
            if stripped == raw or stripped.startswith(raw + "_"):
                agg[i] += shap_vals_pre[0][j]
    shap_vals = _np.array([agg])
    explanation = shap.Explanation(
        values=shap_vals[0],
        base_values=explainer.expected_value,
        data=X_input.iloc[0].values,
        feature_names=raw_feat_names,
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.waterfall_plot(explanation, show=False)
    plt.title("Feature Contributions to Default Risk")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Feature values table
    st.subheader("Input Features")
    feat_df = pd.DataFrame({
        "Feature": deploy["features"],
        "Description": [descs[f] for f in deploy["features"]],
        "Value": [inputs[f] for f in deploy["features"]],
        "SHAP Contribution": shap_vals[0],
    })
    st.dataframe(feat_df, use_container_width=True)

with tab3:
    st.header("Fairness Analysis")
    st.write("Model fairness metrics across sensitive attributes:")

    st.dataframe(fairness_summary.style.format("{:.4f}"), use_container_width=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    fairness_summary.plot(kind="barh", ax=ax)
    ax.axvline(x=0.1, color="red", linestyle="--", label="Threshold (0.1)")
    ax.set_title("Fairness Metrics by Attribute")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with tab4:
    st.header("Data Drift Monitoring")

    if drift_temporal is None:
        st.warning("Drift data not found. Run notebook 06_mlops.ipynb first.")
    else:
        subtab1, subtab2 = st.tabs(["Temporal Drift", "Stress Test"])

        with subtab1:
            st.subheader("Feature Distribution Drift Over Time")
            st.write("PSI measures how much each feature's distribution has shifted from training data.")

            # PSI Heatmap
            pivot = drift_temporal.pivot(index="feature", columns="year", values="psi")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn_r", ax=ax,
                        vmin=0, vmax=0.5, linewidths=0.5,
                        cbar_kws={"label": "PSI"})
            ax.set_title("Population Stability Index by Feature and Year")
            ax.set_ylabel("")
            ax.set_xlabel("")
            st.pyplot(fig)
            plt.close()

            st.caption("PSI < 0.1: stable | 0.1-0.25: moderate drift | > 0.25: significant drift")

            # Distribution overlay selector
            st.subheader("Distribution Comparison")
            col1, col2 = st.columns(2)
            with col1:
                feature = st.selectbox("Feature:", deploy["features"], key="drift_feat")
            with col2:
                year = st.selectbox("Compare year:", ["2016", "2017", "2018"], key="drift_year")

            train_data = drift_samples[drift_samples["period"] == "train"][feature]
            year_data = drift_samples[drift_samples["period"] == year][feature]

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.hist(train_data, bins=50, alpha=0.5, density=True, label="Train (≤2015)", color="steelblue")
            ax.hist(year_data, bins=50, alpha=0.5, density=True, label=f"Test ({year})", color="coral")
            ax.set_title(f"{feature} Distribution: Training vs {year}")
            ax.set_xlabel(feature)
            ax.set_ylabel("Density")
            ax.legend()
            st.pyplot(fig)
            plt.close()

            # Performance trend
            st.subheader("Model Performance Over Time")
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(drift_perf["period"], drift_perf["AUROC"], "o-", label="AUROC", color="steelblue", linewidth=2, markersize=8)
            ax.plot(drift_perf["period"], drift_perf["F1"], "s-", label="F1", color="coral", linewidth=2, markersize=8)
            ax.set_title("EBM (Champion) Performance Over Time")
            ax.set_xlabel("Year")
            ax.set_ylabel("Score")
            ax.legend()
            ax.grid(alpha=0.3)
            st.pyplot(fig)
            plt.close()

            st.dataframe(drift_perf.style.format({
                "AUROC": "{:.4f}", "AUPRC": "{:.4f}", "F1": "{:.4f}",
                "Balanced Accuracy": "{:.4f}", "default_rate": "{:.1%}",
            }), use_container_width=True)

        with subtab2:
            st.subheader("Simulated Drift Stress Test")
            st.write("Shift feature distributions to see how the model responds.")

            ref_sample = drift_samples[drift_samples["period"] == "train"].drop(columns=["period"])

            # Feature shift sliders
            feature_ranges = {
                "int_rate": (-10.0, 10.0, 0.5),
                "term": (-12.0, 12.0, 1.0),
                "acc_open_past_24mths": (-5.0, 5.0, 1.0),
                "annual_inc": (-30000.0, 30000.0, 1000.0),
                "dti": (-15.0, 15.0, 0.5),
                "fico": (-100.0, 100.0, 5.0),
                "loan_amnt": (-10000.0, 10000.0, 500.0),
                "tot_hi_cred_lim": (-50000.0, 50000.0, 5000.0),
                "avg_cur_bal": (-20000.0, 20000.0, 1000.0),
                "total_bc_limit": (-15000.0, 15000.0, 1000.0),
            }

            shifts = {}
            cols = st.columns(2)
            for i, feat in enumerate(deploy["features"]):
                lo, hi, step = feature_ranges.get(feat, (-10.0, 10.0, 1.0))
                with cols[i % 2]:
                    shifts[feat] = st.slider(f"{feat}", lo, hi, 0.0, step, key=f"shift_{feat}")

            # Apply shifts and compute drift
            shifted = simulate_drift(ref_sample, shifts)

            psi_results = {}
            for feat in deploy["features"]:
                psi_results[feat] = compute_psi(ref_sample[feat].values, shifted[feat].values)

            # Alert banner
            max_psi = max(psi_results.values())
            if max_psi > 0.25:
                st.error("DRIFT DETECTED — model retraining recommended")
            elif max_psi > 0.1:
                st.warning("Moderate drift detected — monitor closely")
            else:
                st.success("Model stable — no significant drift detected")

            # PSI table
            psi_df = pd.DataFrame([
                {"Feature": f, "PSI": round(p, 4),
                 "Status": "Significant" if p > 0.25 else "Moderate" if p > 0.1 else "Stable"}
                for f, p in psi_results.items()
            ])
            st.dataframe(psi_df, use_container_width=True)

            # Default rate comparison
            ref_prob = ebm.predict_proba(ref_sample)[:, 1]
            shifted_prob = ebm.predict_proba(shifted)[:, 1]
            ref_default = float((ref_prob >= threshold).mean())
            shifted_default = float((shifted_prob >= threshold).mean())

            col1, col2, col3 = st.columns(3)
            col1.metric("Reference Default Rate", f"{ref_default:.1%}")
            col2.metric("Shifted Default Rate", f"{shifted_default:.1%}")
            col3.metric("Change", f"{shifted_default - ref_default:+.1%}")

            # Distribution overlay for most-shifted feature
            active_shifts = {f: abs(s) for f, s in shifts.items() if s != 0}
            if active_shifts:
                most_shifted_feat = max(active_shifts, key=active_shifts.get)
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.hist(ref_sample[most_shifted_feat], bins=50, alpha=0.5, density=True,
                        label="Reference", color="steelblue")
                ax.hist(shifted[most_shifted_feat], bins=50, alpha=0.5, density=True,
                        label="Shifted", color="coral")
                ax.set_title(f"{most_shifted_feat}: Reference vs Shifted (PSI={psi_results[most_shifted_feat]:.3f})")
                ax.legend()
                st.pyplot(fig)
                plt.close()
