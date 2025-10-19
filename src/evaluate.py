import argparse
import logging
import os

import joblib
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import shap
from scikitplot.metrics import plot_precision_recall, plot_roc

from src.data import load_dataset, time_sorted_split
from src.metrics import (
    expected_cost,
    plot_calibration_curve,
    pr_auc,
    precision_at_k,
    recall_at_fpr,
    roc_auc,
)
from src.threshold import (
    find_best_threshold_by_cost,
    persist_thresholds,
    threshold_at_top_k,
)
from src.utils import load_yaml, logger, timer


@timer
def evaluate_model(config_path: str):
    """Main function to evaluate the trained model."""
    # 1. Load configs, model, and data
    cfg = load_yaml(config_path)
    model_path = f"{cfg['output_dir']}inference.joblib"
    logger.info(f"Loading model from {model_path}")
    pipeline = joblib.load(model_path)

    df = load_dataset(cfg)
    _, _, test_df = time_sorted_split(df, cfg)

    X_test = test_df.drop(columns=[cfg["target_col"]])
    y_test = test_df[cfg["target_col"]]

    # 2. Make predictions
    logger.info("Generating predictions on the test set.")
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

    # 3. Calculate metrics
    metrics = {
        "pr_auc": pr_auc(y_test, y_pred_proba),
        "roc_auc": roc_auc(y_test, y_pred_proba),
        "recall_at_target_fpr": recall_at_fpr(
            y_test.to_numpy(), y_pred_proba, cfg["evaluation"]["target_fpr"]
        ),
        "precision_at_top_k": precision_at_k(
            y_test.to_numpy(), y_pred_proba, k=cfg["evaluation"]["top_k_daily"]
        ),
    }
    logger.info(f"Test Metrics: {metrics}")

    # 4. Find optimal thresholds
    best_threshold_cost, min_cost = find_best_threshold_by_cost(
        y_test.to_numpy(),
        y_pred_proba,
        cost_fp=cfg["evaluation"]["cost_fp"],
        cost_fn=cfg["evaluation"]["cost_fn"],
    )
    metrics["min_expected_cost"] = min_cost

    top_k_threshold = threshold_at_top_k(
        y_pred_proba, k=cfg["evaluation"]["top_k_daily"]
    )

    # For this project, we'll choose the cost-based threshold as the main one
    final_thresholds = {
        "chosen": best_threshold_cost,
        "by_cost": best_threshold_cost,
        "by_topk": top_k_threshold,
    }
    persist_thresholds("configs/thresholds.yaml", final_thresholds)

    # 5. Generate and save plots
    plots_dir = os.path.join(cfg["output_dir"], "plots")
    os.makedirs(plots_dir, exist_ok=True)

    fig_pr = plt.figure()
    plot_precision_recall(y_test, pipeline.predict_proba(X_test), ax=fig_pr.gca())
    pr_curve_path = os.path.join(plots_dir, "pr_curve.png")
    fig_pr.savefig(pr_curve_path)

    fig_roc = plt.figure()
    plot_roc(y_test, pipeline.predict_proba(X_test), ax=fig_roc.gca())
    roc_curve_path = os.path.join(plots_dir, "roc_curve.png")
    fig_roc.savefig(roc_curve_path)

    fig_cal = plot_calibration_curve(y_test.to_numpy(), y_pred_proba)
    cal_curve_path = os.path.join(plots_dir, "calibration_curve.png")
    fig_cal.savefig(cal_curve_path)

    # 6. SHAP Explanations
    logger.info("Generating SHAP explanations.")
    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocessing"]
    X_test_transformed = preprocessor.transform(X_test)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_transformed)

    # Global summary plot
    fig_shap_summary = plt.figure()
    shap.summary_plot(shap_values, X_test_transformed, show=False)
    shap_summary_path = os.path.join(plots_dir, "shap_summary.png")
    plt.savefig(shap_summary_path, bbox_inches="tight")

    # 7. Log to MLflow
    # Find the last run from training to log evaluation results to
    experiment = mlflow.get_experiment_by_name(cfg["mlflow"]["experiment_name"])
    last_run = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"], max_results=1
    ).iloc[0]

    with mlflow.start_run(run_id=last_run.run_id):
        mlflow.log_metrics(metrics)
        mlflow.log_dict(final_thresholds, "thresholds.json")
        mlflow.log_artifacts(plots_dir, artifact_path="plots")

    logger.info("Evaluation complete. All metrics and artifacts logged to MLflow.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the config file.")
    args = parser.parse_args()
    evaluate_model(args.config)
