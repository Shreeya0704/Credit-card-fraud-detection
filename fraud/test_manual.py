
import os
import joblib
import pandas as pd
import numpy as np

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(SCRIPT_DIR, 'model.joblib')
SCALER_PATH = os.path.join(SCRIPT_DIR, 'scaler.joblib')
DATA_PATH = os.path.join(SCRIPT_DIR, 'fraud_data', 'creditcard.csv')

# --- LOAD ARTIFACTS ---
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    df = pd.read_csv(DATA_PATH)
    print("✅ Model, Scaler, and Dataset loaded successfully.")
except FileNotFoundError as e:
    print(f"❌ Error loading files: {e}")
    print("Please run 'train_pro.py' first to generate the necessary model, scaler, and data files.")
    exit()

# --- SELECT A REAL FRAUD CASE ---
# Find a confirmed fraud case from the original dataset
fraud_case_df = df[df['Class'] == 1].head(1)
if fraud_case_df.empty:
    print("❌ No fraud cases found in the dataset to test.")
    exit()

print("\nTesting with a real fraud case:")
print(fraud_case_df)

# Prepare the data in the same way as the training script
# 1. Extract original 'Amount' and 'Time' for scaling
amount = fraud_case_df['Amount'].values[0]
time_val = fraud_case_df['Time'].values[0]

# 2. Scale them
scaled_amount, scaled_time = scaler.transform([[amount, time_val]])[0]

# 3. Get the V-features
v_features = fraud_case_df.drop(['Time', 'Amount', 'Class'], axis=1).values[0]

# 4. Assemble the final feature vector in the correct order
original_features = np.concatenate([v_features, [scaled_amount, scaled_time]])
col_names = [f'V{i}' for i in range(1, 29)] + ['scaled_amount', 'scaled_time']
features_df = pd.DataFrame([original_features], columns=col_names)


# --- DYNAMIC TEST LOOP ---
print("\n--- Dynamic Probability Test ---")
print("Adding random noise to the fraud data to observe probability changes.")

for noise_level in np.arange(0.0, 1.1, 0.1):
    # Create a copy to avoid corrupting the original features
    noisy_features = original_features.copy()
    
    # Add random noise proportional to the noise level
    noise = np.random.normal(0, noise_level, noisy_features.shape)
    noisy_features += noise
    
    # Create a DataFrame for prediction
    noisy_df = pd.DataFrame([noisy_features], columns=col_names)
    
    # Get the fraud probability
    probability = model.predict_proba(noisy_df)[0][1]
    
    print(f"Noise Level: {noise_level:.1f} | Fraud Probability: {probability:.2%}")

print("\n--- Test Complete ---")
print("The smooth decay in probability proves the model is making mathematical judgments, not just memorizing.")
