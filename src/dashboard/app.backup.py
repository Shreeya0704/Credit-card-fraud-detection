import joblib
import logging
import pandas as pd
import streamlit as st
import shap
import numpy as np
from PIL import Image

from src.utils import load_yaml, logger
from src.metrics import precision_recall_curve, expected_cost

# --- Page Config ---
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🕵️",
    layout="wide",
)

# --- Load Model & Artifacts ---
@st.cache_resource
def load_model_and_artifacts():
    """Load the inference pipeline, thresholds, and plots."""
    try:
        model = joblib.load("models/inference.joblib")
        thresholds = load_yaml("configs/thresholds.yaml")
        pr_curve_img = Image.open("models/plots/pr_curve.png")
        shap_summary_img = Image.open("models/plots/shap_summary.png")
        return model, thresholds, pr_curve_img, shap_summary_img
    except FileNotFoundError as e:
        st.error(f"Could not load model or artifacts. Please run the training and evaluation pipeline first. Missing: {e.filename}")
        return None, None, None, None

MODEL, THRESHOLDS, PR_CURVE_IMG, SHAP_SUMMARY_IMG = load_model_and_artifacts()

# --- App Layout ---
st.title("🕵️ Credit Card Fraud Detection Dashboard")

if not MODEL:
    st.stop()

# --- Sidebar ---
st.sidebar.header("Controls")

# Threshold slider
chosen_threshold = THRESHOLDS.get("chosen", 0.5)
threshold = st.sidebar.slider(
    "Probability Threshold",
    min_value=0.0,
    max_value=1.0,
    value=chosen_threshold,
    step=0.01,
    help="Adjust the cutoff for classifying a transaction as fraudulent.",
)

# File uploader
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader(
    "Upload a CSV file for scoring", type=["csv"]
)

# --- Main Content ---
if uploaded_file is not None:
    st.header("Scoring Results")
    try:
        df = pd.read_csv(uploaded_file)
        
        # Ensure target column is present for metrics calculation
        if "Class" not in df.columns:
            st.warning("The uploaded CSV does not contain a 'Class' column. Metrics cannot be calculated.")
            y_true = None
        else:
            y_true = df["Class"]

        # Score data
        with st.spinner("Scoring data and calculating explanations..."):
            scores = MODEL.predict_proba(df)[:, 1]
            df["score"] = scores
            df["is_fraud"] = (df["score"] >= threshold).astype(int)

            # SHAP Explanations (for the top 3 reasons)
            preprocessor = MODEL.named_steps["preprocessing"]
            df_transformed = preprocessor.transform(df)
            explainer = shap.TreeExplainer(MODEL.named_steps["model"])
            shap_values = explainer.shap_values(df_transformed)[1]
            
            feature_names = df_transformed.columns
            top_reasons = []
            for i in range(len(shap_values)):
                top_indices = np.argsort(np.abs(shap_values[i]))[-3:][::-1]
                reasons = f"{feature_names[top_indices[0]]} ({shap_values[i, top_indices[0]]:.2f}), {feature_names[top_indices[1]]} ({shap_values[i, top_indices[1]]:.2f}), {feature_names[top_indices[2]]} ({shap_values[i, top_indices[2]]:.2f})"
                top_reasons.append(reasons)
            df["reasons"] = top_reasons

        # --- Display Metrics ---
        st.subheader("Live Metrics at Selected Threshold")
        col1, col2, col3, col4 = st.columns(4)
        
        if y_true is not None:
            y_pred = df["is_fraud"]
            precision, recall, _ = precision_recall_curve(y_true, scores)
            
            # Find precision and recall at the current threshold
            idx = np.searchsorted(np.unique(scores), threshold)
            current_precision = precision[idx] if idx < len(precision) else 0
            current_recall = recall[idx] if idx < len(recall) else 0

            cost = expected_cost(y_true, scores, 100, 10000, threshold)

            col1.metric("Precision", f"{current_precision:.2%}")
            col2.metric("Recall", f"{current_recall:.2%}")
            col3.metric("False Positives", y_pred[y_true == 0].sum())
            col4.metric("Expected Cost", f"${cost:,.0f}")

        # --- Display Results Table ---
        st.subheader("Scored Transactions")
        st.dataframe(df[["Time", "Amount", "score", "is_fraud", "reasons"]].sort_values(by="score", ascending=False), use_container_width=True)

    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")

else:
    st.info("Upload a CSV file via the sidebar to begin scoring.")

# --- Static Plots ---
st.header("Model Performance Overview")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Precision-Recall Curve")
    if PR_CURVE_IMG:
        st.image(PR_CURVE_IMG, use_column_width=True)
    else:
        st.warning("PR curve plot not found.")

with col2:
    st.subheader("Global SHAP Summary")
    if SHAP_SUMMARY_IMG:
        st.image(SHAP_SUMMARY_IMG, use_column_width=True)
    else:
        st.warning("SHAP summary plot not found.")
