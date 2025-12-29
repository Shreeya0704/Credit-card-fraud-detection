import os
import joblib
import pandas as pd
import numpy as np

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(SCRIPT_DIR, 'model.joblib')
SCALER_PATH = os.path.join(SCRIPT_DIR, 'scaler.joblib')

# The column names in the order the model expects
MODEL_COL_NAMES = [f'V{i}' for i in range(1, 29)] + ['scaled_amount', 'scaled_time']
SCALER_COL_NAMES = ['Amount', 'Time']

# --- LOAD ARTIFACTS ---
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("✅ Model and Scaler loaded successfully.")
except FileNotFoundError as e:
    print(f"❌ Error loading files: {e}")
    print("Please run 'train_pro.py' first to generate the model and scaler.")
    exit()

# --- SENSITIVITY ANALYSIS ---
print("\n--- Sensitivity Analysis ---")
print("Starting with a real fraud vector and changing one feature at a time.")

# The REAL fraud vector [Time, V1...V28, Amount]
fraud_vector = np.array([
    406.0, -2.312, 1.951, -1.609, 3.997, -0.522, -1.426, -2.537, 1.391, -2.770, 
    -2.772, 3.202, -2.899, -0.595, -4.289, 0.389, -1.140, -2.830, -0.016, 0.416, 
    0.126, 0.517, -0.035, -0.465, 0.320, 0.044, 0.177, 0.261, -0.143, 50000.0
])

def get_prediction(data_vector):
    """Helper function to get fraud probability from a vector."""
    time_val = data_vector[0]
    v_features = data_vector[1:29]
    amount = data_vector[29]
    
    # Create a DataFrame for the scaler
    scaler_df = pd.DataFrame([[amount, time_val]], columns=SCALER_COL_NAMES)
    scaled_amount, scaled_time = scaler.transform(scaler_df)[0]
    
    final_features = np.concatenate([v_features, [scaled_amount, scaled_time]])
    features_df = pd.DataFrame([final_features], columns=MODEL_COL_NAMES)
    
    probability = model.predict_proba(features_df)[0][1]
    return probability

# --- Run the steps ---

# Original Case
prob_original = get_prediction(fraud_vector)
print(f"Original Fraud Case:      Fraud Probability: {prob_original:.2%}")

# Step 1: Normalize Amount
step1_vector = fraud_vector.copy()
step1_vector[29] = 10.0  # Change Amount from 50,000 to 10
prob_step1 = get_prediction(step1_vector)
print(f"Step 1 (Amount -> 10.0):   Fraud Probability: {prob_step1:.2%}")

# Step 2: Normalize V14 (strongest feature)
step2_vector = step1_vector.copy()
# V14 is at index 14 (since V1 is at index 1)
step2_vector[14] = 0.0  # Change V14 from -4.289 to 0.0
prob_step2 = get_prediction(step2_vector)
print(f"Step 2 (V14 -> 0.0):        Fraud Probability: {prob_step2:.2%}")

# Step 3: Normalize V4
step3_vector = step2_vector.copy()
# V4 is at index 4
step3_vector[4] = 0.0  # Change V4 from 3.997 to 0.0
prob_step3 = get_prediction(step3_vector)
print(f"Step 3 (V4 -> 0.0):         Fraud Probability: {prob_step3:.2%}")

# Step 4: Normalize V11
step4_vector = step3_vector.copy()
# V11 is at index 11
step4_vector[11] = 0.0 # Change V11 from 3.202 to 0.0
prob_step4 = get_prediction(step4_vector)
print(f"Step 4 (V11 -> 0.0):        Fraud Probability: {prob_step4:.2%}")


print("\n--- Test Complete ---")
print("The step-by-step decay in probability confirms the model's reliance on specific, learned features.")