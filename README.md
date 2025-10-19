# End-to-End Credit Card Fraud Detection System

This project provides a complete, production-ready machine learning system for detecting fraudulent credit card transactions. It includes a reproducible, config-driven pipeline for data preprocessing, feature engineering, model training, and evaluation, with a focus on time-aware validation and cost-sensitive thresholding.

The system is built to work out-of-the-box with the Kaggle Credit Card Fraud dataset but is flexible enough to handle richer data schemas.

## Project Overview

- **Objective**: Train a cost-sensitive, leakage-safe fraud classifier using classic ML (LightGBM/XGBoost).
- **Evaluation**: Employs time-aware evaluation with strong, practical metrics (PR-AUC, Recall@FPR) and chooses an optimal prediction threshold based on expected cost and top-K review capacity.
- **Explainability**: Integrates SHAP for both global and local model explanations.
- **Deployment**: Includes a deployable FastAPI service for real-time scoring and a simple Streamlit dashboard for analyst triage.
- **Reproducibility**: Uses MLflow for experiment tracking and a configuration-driven approach for the entire pipeline.

## Dataset Setup

1.  Download the dataset from [Kaggle: Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
2.  Place the downloaded `creditcard.csv` file into the `data/raw/` directory. The final path should be `data/raw/creditcard.csv`.

## Quickstart

1.  **Create Virtual Environment & Activate:**
    ```bash
    # For macOS/Linux
    python3 -m venv .venv
    source .venv/bin/activate

    # For Windows
    python -m venv .venv
    .\.venv\Scripts\activate
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Training Pipeline:**
    This script preprocesses data, trains the model, and logs all artifacts to MLflow.
    ```bash
    bash scripts/run_train.sh
    ```

4.  **Run the Evaluation Pipeline:**
    This script evaluates the best model on the test set, calculates final metrics, determines the optimal threshold, and saves plots and SHAP explanations.
    ```bash
    bash scripts/run_eval.sh
    ```

5.  **Serve the API:**
    This starts the FastAPI service for real-time predictions.
    ```bash
    bash scripts/serve_api.sh
    ```
    You can then send a `POST` request to `http://127.0.0.1:8000/score` with transaction data.

6.  **Launch the Dashboard:**
    This runs the Streamlit application for interactive analysis.
    ```bash
    bash scripts/run_dashboard.sh
    ```

## Important Notes

### CPU-Only
This project is designed for **CPU-only** environments. No GPUs or related libraries (CUDA, etc.) are required.

### Prediction Thresholding
A key part of this system is choosing a probability threshold that converts model scores into fraud/not-fraud labels. This project implements two strategies:
1.  **Expected Cost Minimization**: Finds the threshold that minimizes a cost function defined by the cost of a false positive (e.g., investigating a legitimate transaction) and a false negative (e.g., missing a fraudulent transaction).
2.  **Top-K Capacity**: Finds the threshold that flags the top `K` highest-risk transactions per day for manual review.

The final chosen threshold is persisted in `configs/thresholds.yaml` after evaluation.

### Richer Feature Schemas
The feature engineering pipeline (`src/features.py`) can automatically handle richer datasets. If columns like `card_id`, `merchant_id`, or `country` are present in the input CSV, the system will generate additional features like transaction counts and target-encoded categories. If they are absent, it gracefully falls back to using only the base Kaggle features.

### MLflow Usage
This project uses a local MLflow instance to store experiment runs in the `mlruns/` directory. To view the results:
1.  Run the UI server:
    ```bash
    mlflow ui
    ```
2.  Navigate to `http://127.0.0.1:5000` in your browser to compare runs, view metrics (PR-AUC, ROC-AUC), and inspect saved artifacts (plots, models).

### Common Pitfalls Addressed
- **Class Imbalance**: Handled using `scale_pos_weight` in the tree-based models, which is generally more robust and computationally cheaper than oversampling methods like SMOTE.
- **Model Calibration**: The evaluation script generates a calibration curve to check if the model's predicted probabilities are reliable.
- **Data Leakage**: The project strictly uses time-sorted data splits and a sequential feature generator to ensure that the model never learns from future information.
