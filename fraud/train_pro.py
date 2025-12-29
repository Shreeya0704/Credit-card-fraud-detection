import os
import kaggle
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# Define paths relative to the script's location
SCRIPT_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(SCRIPT_DIR, 'fraud_data')
CSV_FILE_PATH = os.path.join(DATA_PATH, 'creditcard.csv')
METRICS_FILE_PATH = os.path.join(SCRIPT_DIR, 'metrics.txt')
CONFUSION_MATRIX_PATH = os.path.join(SCRIPT_DIR, 'confusion_matrix.png')
FEATURE_IMPORTANCE_PATH = os.path.join(SCRIPT_DIR, 'feature_importance.png')
MODEL_PATH = os.path.join(SCRIPT_DIR, 'model.joblib')
SCALER_PATH = os.path.join(SCRIPT_DIR, 'scaler.joblib')


print("Starting Professional XGBoost Model Training...")

# 1. Data Loading
if not os.path.exists(CSV_FILE_PATH):
    print("Dataset not found locally. Downloading from Kaggle...")
    print("Ensure you have a 'kaggle.json' file in your ~/.kaggle/ directory for authentication.")
    kaggle.api.authenticate()
    os.makedirs(DATA_PATH, exist_ok=True) # Ensure data directory exists
    kaggle.api.dataset_download_files('mlg-ulb/creditcardfraud', path=DATA_PATH, unzip=True)
    print("Dataset downloaded successfully.")
else:
    print("Dataset found locally. Skipping download.")

df = pd.read_csv(CSV_FILE_PATH)

# 2. Preprocessing
print("Performing advanced preprocessing...")
# The scaler should be fitted on the columns in the order they will be transformed.
scaler = StandardScaler()
df[['scaled_amount', 'scaled_time']] = scaler.fit_transform(df[['Amount', 'Time']])

# Save the scaler IMMEDIATELY after fitting
joblib.dump(scaler, SCALER_PATH)

df.drop(['Time', 'Amount'], axis=1, inplace=True) # Drop original columns after scaling

X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("Data split into training and testing sets.")

# 3. Calculate Class Imbalance Ratio for Weighting
ratio = float(np.sum(y_train == 0)) / np.sum(y_train == 1)
print(f"Calculated scale_pos_weight for class imbalance: {ratio:.2f}")

# 4. Train High-Performance XGBoost Model
print("Training XGBoost model with optimized parameters...")
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=ratio,
    n_jobs=1,  # Critical for preventing hangs in some environments
    tree_method="hist",
    random_state=42
)

model.fit(X_train, y_train)

# 5. Evaluation and Artifact Generation
print("Evaluating model and generating artifacts...")
y_pred = model.predict(X_test)

# Save classification report
with open(METRICS_FILE_PATH, 'w') as f:
    f.write(classification_report(y_test, y_pred))

# Generate and save confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Fraud', 'Fraud'], yticklabels=['Non-Fraud', 'Fraud'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig(CONFUSION_MATRIX_PATH)

# Generate Feature Importance Plot
plt.figure(figsize=(10, 8))
xgb.plot_importance(model, max_num_features=15, height=0.5)
plt.title("What actually indicates Fraud? (Top 15 Features)")
plt.tight_layout()
plt.savefig(FEATURE_IMPORTANCE_PATH)

# 6. Save the Model
joblib.dump(model, MODEL_PATH)

print("\nXGBoost Training Complete.")
print(f"Metrics saved to {METRICS_FILE_PATH}")
print(f"Confusion matrix saved to {CONFUSION_MATRIX_PATH}")
print(f"Feature Importance saved to {FEATURE_IMPORTANCE_PATH}")
print(f"Scaler saved to {SCALER_PATH}")