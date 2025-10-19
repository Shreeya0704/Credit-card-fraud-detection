import joblib
import logging
import pandas as pd
import numpy as np
import shap
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional

from src.utils import load_yaml, logger

# 1. Initialize FastAPI app
app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="API for scoring transactions for fraud risk.",
    version="0.1.0",
)

# 2. Load model and thresholds at startup
MODEL = None
THRESHOLDS = None
EXPLAINER = None

@app.on_event("startup")
def load_model_and_config():
    """Load the inference pipeline and configuration at application startup."""
    global MODEL, THRESHOLDS, EXPLAINER
    try:
        MODEL = joblib.load("models/inference.joblib")
        THRESHOLDS = load_yaml("configs/thresholds.yaml")
        logger.info("Model and thresholds loaded successfully.")

        # Initialize SHAP explainer
        model_component = MODEL.named_steps["model"]
        EXPLAINER = shap.TreeExplainer(model_component)
        logger.info("SHAP TreeExplainer initialized.")

    except FileNotFoundError as e:
        logger.error(f"Error loading model/config at startup: {e}")
        MODEL = None # Ensure app starts even if model is missing
        THRESHOLDS = None
        EXPLAINER = None

# 3. Define health check endpoint
@app.get("/health", tags=["Health"])
def health_check():
    """Check the health of the API, including model loading status."""
    if MODEL and THRESHOLDS and EXPLAINER:
        return {"status": "ok", "message": "API is healthy and model is loaded."}
    else:
        raise HTTPException(
            status_code=503,
            detail="Service Unavailable: Model or config not loaded.",
        )

# 4. Define Pydantic model for input validation
class Transaction(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float
    # Optional richer features
    card_id: Optional[str] = None
    merchant_id: Optional[str] = None
    country: Optional[str] = None
    mcc: Optional[str] = None

class ScoringRequest(BaseModel):
    transactions: List[Transaction]

class Reason(BaseModel):
    feature: str
    contribution: float
    note: Optional[str] = None

class ScoringResponse(BaseModel):
    score: float
    is_fraud: bool
    reasons: List[Reason] = Field(default_factory=list)

# 5. Define scoring endpoint
@app.post("/score", response_model=List[ScoringResponse], tags=["Scoring"])
def score_transactions(request: ScoringRequest):
    """Score a list of transactions for fraud risk."""
    if not MODEL or not THRESHOLDS or not EXPLAINER:
        raise HTTPException(
            status_code=503, detail="Model is not loaded. Cannot perform scoring."
        )

    try:
        responses = []
        for transaction in request.transactions:
            # build DataFrame from payload (no label)
            X_df = pd.DataFrame([transaction.dict()])
            if "Class" in X_df.columns:
                X_df = X_df.drop(columns=["Class"])
            # best-effort numeric cast (CSV/JSON often come as strings)
            for c in X_df.columns:
                X_df[c] = pd.to_numeric(X_df[c], errors="ignore")

            # ---- predict proba (robust to shape) ----
            proba = MODEL.predict_proba(X_df)
            proba = np.asarray(proba)
            if proba.ndim == 2 and proba.shape[1] >= 2:
                score = float(proba[0, 1])
            elif proba.ndim == 2 and proba.shape[1] == 1:
                score = float(proba[0, 0])
            else:
                score = float(np.ravel(proba)[0])

            chosen_threshold = THRESHOLDS.get("chosen", 0.5)
            is_fraud = bool(score >= chosen_threshold)

            # ---- SHAP reasons on transformed features ----
            reasons = []
            try:
                PREPROCESSOR = MODEL.named_steps["preprocessing"]
                
                if PREPROCESSOR is not None:
                    Xt = PREPROCESSOR.transform(X_df)
                else:
                    Xt = X_df
                # to numpy for SHAP
                from scipy import sparse
                Xt_np = Xt.toarray() if sparse.issparse(Xt) else np.asarray(Xt)

                vals = EXPLAINER.shap_values(Xt_np, check_additivity=False)
                # list-of-arrays (classes) vs single array
                if isinstance(vals, list):
                    sv_row = vals[1][0] if len(vals) > 1 else vals[0][0]
                else:
                    sv_row = vals[0]

                # feature names from preprocessor if available; else use Kaggle schema names
                try:
                    feature_names = PREPROCESSOR.get_feature_names_out().tolist()
                except Exception:
                    expected = [f"V{i}" for i in range(1, 29)] + ["amount_log", "time_sin", "time_cos"]
                    feature_names = (
                        expected if Xt_np.shape[1] == len(expected)
                        else (X_df.columns.tolist() if Xt_np.shape[1] == len(X_df.columns)
                              else [f"f{i}" for i in range(Xt_np.shape[1])])
                    )

                top_idx = np.argsort(np.abs(sv_row))[::-1][:5]
                reasons = [
                    Reason(feature=feature_names[i] if i < len(feature_names) else f"f{i}",
                           contribution=float(sv_row[i]))
                    for i in top_idx
                ]
            except Exception as e:
                logger.error("SHAP reasons failed: %s", e, exc_info=True)
                reasons = [Reason(feature="unavailable", contribution=0.0, note=str(e))]
            
            responses.append(
                ScoringResponse(score=score, is_fraud=is_fraud, reasons=reasons)
            )
        
        return responses

    except Exception as e:
        logger.error(f"An error occurred during scoring: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")
