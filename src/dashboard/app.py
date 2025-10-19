import numpy as np
import pandas as pd
import joblib
import shap
from pathlib import Path
from scipy import sparse
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
import yaml

st.set_page_config(page_title="Credit Card Fraud Detection Dashboard", layout="wide")

# ---------- SPEED KNOBS ----------
MAX_ROWS = 5000        # cap rows scored in UI
MAX_SHAP = 200         # explain top-N riskiest only

# ---------- CACHING ----------
@st.cache_resource
def load_pipeline():
    pipe = joblib.load("models/inference.joblib")
    if isinstance(pipe, Pipeline) and len(pipe.steps) >= 2:
        pre = Pipeline(pipe.steps[:-1])
        model = pipe.steps[-1][1]
    else:
        pre, model = None, pipe
    # TreeExplainer for tree models; fallback to None
    try:
        booster = getattr(model, "booster_", model)
        expl = shap.TreeExplainer(booster)
    except Exception:
        expl = None
    return pipe, pre, model, expl

@st.cache_data
def read_csv_cached(csv_file):
    try:
        return pd.read_csv(csv_file, engine="pyarrow")
    except Exception:
        return pd.read_csv(csv_file)

def to_numpy(x):
    if sparse.issparse(x):
        return x.toarray()
    return np.asarray(x)

def feature_names_from_pre(pre, X_df, Xt_np):
    if pre is not None:
        try:
            return pre.get_feature_names_out().tolist()
        except Exception:
            pass
    # Fallbacks
    if Xt_np.shape[1] == len(X_df.columns):
        return X_df.columns.tolist()
    expected = [f"V{i}" for i in range(1, 29)] + ["amount_log", "time_sin", "time_cos"]
    return expected if Xt_np.shape[1] == len(expected) else [f"f{i}" for i in range(Xt_np.shape[1])]

# ---------- UI ----------
st.sidebar.header("Controls")
import yaml

def _default_thr():
    try:
        d = yaml.safe_load(open("configs/thresholds.yaml", "r"))
        chosen = d.get("chosen", "by_cost")
        return float(d.get(chosen, {}).get("threshold", 0.5))
    except Exception:
        return 0.5

thr = st.sidebar.slider(
    "Probability Threshold",
    0.0, 1.0,
    value=_default_thr(),
    step=0.01
)
uploaded = st.sidebar.file_uploader("Upload a CSV file for scoring", type=["csv"])

st.title("🕵️ Credit Card Fraud Detection Dashboard")

# Show saved training plots (avoid recompute)
plots_dir = Path("models/plots")
if plots_dir.exists():
    cols = st.columns(3)
    imgs = sorted(plots_dir.glob("*.png"))[:3]
    for i, p in enumerate(imgs):
        cols[i % 3].image(str(p), caption=p.name, use_container_width=True)

pipe, PRE, MODEL, EXPL = load_pipeline()

if uploaded:
    df = read_csv_cached(uploaded)
    if "Class" in df.columns:
        df = df.drop(columns=["Class"])
    # Cap rows for speed
    if len(df) > MAX_ROWS:
        df = df.head(MAX_ROWS)

    # Predict
    probs = pipe.predict_proba(df)
    probs = np.asarray(probs)
    scores = probs[:, 1] if probs.ndim == 2 and probs.shape[1] >= 2 else np.ravel(probs)
    preds = (scores >= thr).astype(int)

    st.subheader("Scoring Summary")
    st.write(f"Rows scored: **{len(df):,}**  |  Threshold: **{thr:.3f}**  |  Positives flagged: **{preds.sum():,}**")

    out = df.copy()
    out["score"] = scores
    out["is_fraud"] = preds
    st.dataframe(out.sort_values("score", ascending=False).head(20), use_container_width=True)

    # SHAP on top-N only
    st.subheader("Top-N SHAP Explanations")
    try:
        if EXPL is None:
            st.info("SHAP unavailable for this model.")
        else:
            Xt = PRE.transform(df) if PRE is not None else df
            Xt_np = to_numpy(Xt)
            feat_names = feature_names_from_pre(PRE, df, Xt_np)
            top_idx = np.argsort(scores)[-min(MAX_SHAP, len(df)):]
            vals = EXPL.shap_values(Xt_np[top_idx], check_additivity=False)
            if isinstance(vals, list):  # multi-class style
                vals = vals[1] if len(vals) > 1 else vals[0]
            shap.summary_plot(vals, Xt_np[top_idx], feature_names=feat_names, show=False)
            st.pyplot(plt.gcf(), use_container_width=True)
    except Exception as e:
        st.warning(f"SHAP explanation failed: {e}")
else:
    st.info("Upload a CSV to begin scoring. Saved training plots (if any) are shown above.")