import argparse
import logging

import joblib
import lightgbm as lgb
import mlflow
import numpy as np
import optuna
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline

from src.data import load_dataset, time_sorted_split
from src.features import build_preprocess_pipeline
from src.utils import load_yaml, logger, seed_everything, timer


@timer
def train_model(config_path: str):
    """Main function to train the fraud detection model."""
    # 1. Load configs and setup
    cfg = load_yaml(config_path)
    seed_everything(cfg["random_state"])
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    # 2. Load and split data
    df = load_dataset(cfg)
    train_df, valid_df, _ = time_sorted_split(df, cfg)

    # 3. Build preprocessing pipeline and prepare data
    X_train = train_df.drop(columns=[cfg["target_col"]])
    y_train = train_df[cfg["target_col"]]
    X_valid = valid_df.drop(columns=[cfg["target_col"]])
    y_valid = valid_df[cfg["target_col"]]

    pipeline = build_preprocess_pipeline(cfg, X_train)
    X_train_transformed = pipeline.fit_transform(X_train, y_train)
    X_valid_transformed = pipeline.transform(X_valid)

    # 4. Handle class imbalance
    if cfg["imbalance"].get("use_smote", False):
        logger.info("Using SMOTE for imbalance handling.")
        smote_strategy = cfg["imbalance"].get("smote_strategy", 0.1)
        smote = SMOTE(sampling_strategy=smote_strategy, random_state=cfg["random_state"])
        X_train_transformed, y_train = smote.fit_resample(X_train_transformed, y_train)
        logger.info(f"Data shape after SMOTE: {X_train_transformed.shape}")

    scale_pos_weight = (
        (len(y_train) - np.sum(y_train)) / np.sum(y_train)
        if cfg["modeling"]["class_weight"] == "balanced"
        else 1
    )

    # 5. Hyperparameter tuning with Optuna
    def objective(trial):
        params = {
            "objective": "binary",
            "metric": "aucpr",
            "verbosity": -1,
            "n_jobs": cfg["modeling"]["n_jobs"],
            "seed": cfg["random_state"],
            "scale_pos_weight": scale_pos_weight,
            "num_leaves": trial.suggest_int("num_leaves", 20, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
            "n_estimators": trial.suggest_int("n_estimators", 100, 2000),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train_transformed,
            y_train,
            eval_set=[(X_valid_transformed, y_valid)],
            eval_metric="average_precision",
            callbacks=[
                lgb.early_stopping(
                    stopping_rounds=cfg["modeling"]["early_stopping_rounds"],
                    verbose=False,
                )
            ],
        )

        preds = model.predict_proba(X_valid_transformed)[:, 1]
        pr_auc = average_precision_score(y_valid, preds)
        return pr_auc

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=cfg["modeling"]["optuna_trials"])

    # 6. Train final model with best params and log to MLflow
    with mlflow.start_run() as run:
        logger.info("Training final model with best parameters.")
        best_params = study.best_params
        best_params.update(
            {
                "objective": "binary",
                "metric": "aucpr",
                "verbosity": -1,
                "n_jobs": cfg["modeling"]["n_jobs"],
                "seed": cfg["random_state"],
                "scale_pos_weight": scale_pos_weight,
            }
        )

        final_model = lgb.LGBMClassifier(**best_params)
        final_model.fit(X_train_transformed, y_train)

        # Log to MLflow
        mlflow.log_params(best_params)
        mlflow.log_params(cfg)
        mlflow.log_metric("best_valid_pr_auc", study.best_value)

        # Create and save the full inference pipeline
        inference_pipeline = Pipeline([("preprocessing", pipeline), ("model", final_model)])
        output_path = f"{cfg['output_dir']}inference.joblib"
        joblib.dump(inference_pipeline, output_path)
        mlflow.log_artifact(output_path)

        logger.info(f"Best validation PR-AUC: {study.best_value:.4f}")
        logger.info(f"Final model and pipeline saved to {output_path}")
        logger.info(f"MLflow Run ID: {run.info.run_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the config file.")
    args = parser.parse_args()
    train_model(args.config)
