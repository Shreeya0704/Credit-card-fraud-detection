#  Global Bank Fraud Detection System

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

##  Project Overview
This project is a comprehensive Machine Learning solution designed to detect fraudulent credit card transactions in real-time. Unlike standard classification models, this system integrates **Sensitivity Analysis** to explain *why* a transaction was flagged, providing transparency alongside high accuracy.

The model is built using **XGBoost** to handle class imbalance effectively, ensuring that rare fraud cases are detected with high precision.

##  Key Features
* **High-Performance Classification:** Utilizes XGBoost with weighted parameters to handle the severe class imbalance inherent in fraud datasets.
* **Real-Time Sensitivity Analysis:** A custom "Kill Switch" logic that analyzes feature importance dynamically. It allows the system to determine exactly which variable (e.g., `V4`, `V14`) triggered the fraud alert.
* **Interactive Dashboard:** A user-friendly Streamlit interface for bank analysts to monitor transaction streams.
* **GenAI Integration:** Includes a module (`genai_service.py`) for generating natural language explanations for flagged transactions.
* **Manual Testing Suite:** Dedicated testing scripts to verify model resilience against specific attack vectors.

##  Tech Stack
* **Language:** Python
* **Machine Learning:** Scikit-Learn, XGBoost, Pandas, NumPy
* **Visualization & UI:** Streamlit
* **Version Control:** Git & GitHub

##  Project Structure
```text
├── fraud/
│   ├── app.py                 # Main Streamlit Dashboard application
│   ├── train_real.py          # Training pipeline (Data preprocessing + Model fitting)
│   ├── test_sensitivity.py    # Logic for feature importance & sensitivity checks
│   ├── genai_service.py       # AI service for explaining fraud decisions
│   ├── dashboard.py           # UI Component logic
│   └── test_manual.py         # Unit tests for manual transaction entry
├── .gitignore                 # Files excluded from version control
└── README.md                  # Project Documentation
```
##  Installation & Usage
1. Clone the Repository
```bash
git clone https://github.com/Shreeya0704/Credit-card-fraud-detection.git
cd Credit-card-fraud-detection
```
2. Install Dependencies
```bash
pip install -r requirements.txt
```
(Note: Ensure you have streamlit, xgboost, scikit-learn, and pandas installed)

3. Run the Application
```bash
streamlit run fraud/dashboard.py
```
 Methodology
Data Preprocessing: Standard scaling applied to PCA-transformed features (V1 to V28) and the Amount feature.

Model Training: Trained on historical transaction data using XGBoost. The model was tuned to prioritize Recall (catching as many fraud cases as possible).

Explainability: The system runs a perturbation analysis on high-probability fraud cases to confirm which features are driving the decision (Sensitivity Analysis).

