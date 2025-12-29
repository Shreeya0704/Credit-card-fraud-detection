
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd
import os
from genai_service import ComplianceAgent

app = FastAPI()

# Get the directory of the current script
SCRIPT_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(SCRIPT_DIR, 'model.joblib')
SCALER_PATH = os.path.join(SCRIPT_DIR, 'scaler.joblib')

# Define column names for clarity
MODEL_COL_NAMES = [f'V{i}' for i in range(1, 29)] + ['scaled_amount', 'scaled_time']
SCALER_COL_NAMES = ['Amount', 'Time']

# Load model and scaler at startup
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("✅ Model and Scaler loaded successfully.")
except FileNotFoundError as e:
    model = None
    scaler = None
    print(f"❌ Error loading model/scaler: {e}. Predictions will not be available.")


class PredictionRequest(BaseModel):
    transaction_details: str
    data: List[float]

@app.get("/")
def read_root():
    return {"message": "GenAI Fraud Detection API"}

@app.post('/predict')
def predict(request: PredictionRequest):
    if model is None or scaler is None:
        return {"error": "Model or scaler not loaded. Please ensure training is complete and files are in the 'fraud' directory."}

    # 1. Extract features from the input list [Time, V1...V28, Amount]
    input_list = request.data
    time_val = input_list[0]
    v_features = input_list[1:29]
    amount = input_list[29]

    # 2. Create DataFrame for scaler and transform
    scaler_df = pd.DataFrame([[amount, time_val]], columns=SCALER_COL_NAMES)
    scaled_amount, scaled_time = scaler.transform(scaler_df)[0]
    
    # 3. Reassemble the final feature list for the model
    final_features = v_features + [scaled_amount, scaled_time]
    
    # 4. Create DataFrame for the model with correct column names
    features_df = pd.DataFrame([final_features], columns=MODEL_COL_NAMES)
    
    # 5. Get prediction probability
    probability = model.predict_proba(features_df)[0][1]

    # 6. Apply prediction logic
    is_fraud = False
    if probability > 0.5 or "suspicious" in request.transaction_details.lower():
        is_fraud = True
        
    response = {
        "prediction": 1 if is_fraud else 0,
        "probability": float(probability)
    }

    if is_fraud:
        sanitized_details = ComplianceAgent.redact_pii(request.transaction_details)
        explanation = ComplianceAgent.generate_sar(sanitized_details, probability)
        response["explanation"] = explanation

    return response
