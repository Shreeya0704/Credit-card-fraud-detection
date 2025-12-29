
import os
import kaggle
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Define the path for the data
DATA_PATH = 'fraud_data'
CSV_FILE_PATH = os.path.join(DATA_PATH, 'creditcard.csv')

print("Starting model training process...")

# 1. Data Loading
if not os.path.exists(CSV_FILE_PATH):
    print("Dataset not found locally. Downloading from Kaggle...")
    print("Ensure you have a 'kaggle.json' file in your ~/.kaggle/ directory for authentication.")
    
    # Authenticate with Kaggle API
    kaggle.api.authenticate()
    
    # Download and unzip the dataset
    kaggle.api.dataset_download_files('mlg-ulb/creditcardfraud', path=DATA_PATH, unzip=True)
    print("Dataset downloaded successfully.")
else:
    print("Dataset found locally. Skipping download.")

# Load the dataset
df = pd.read_csv(CSV_FILE_PATH)

# Use a small subset for speed (10% of data)
df = df.sample(frac=0.1, random_state=42).reset_index(drop=True)
print(f"Using a subset of {len(df)} samples for training.")

# Prepare data for RandomForestClassifier
X = df.drop('Class', axis=1)
y = df['Class']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Training RandomForestClassifier model...")
# Initialize and train the RandomForestClassifier
model = RandomForestClassifier(n_estimators=10, n_jobs=-1, random_state=42)
model.fit(X_train, y_train)

# Save the trained model to the 'fraud' directory
model_path = os.path.join(os.path.dirname(__file__), 'model.joblib')
joblib.dump(model, model_path)

print(f"Training Complete. Model Saved to {model_path}")
