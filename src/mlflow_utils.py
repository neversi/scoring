"""MLflow experiment tracking and logging utilities."""

import os
import tempfile
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from src.evaluate import compute_metrics, bootstrap_metrics
from src.config import CONFIG
from src.threshold import (
    threshold_cost_minimization,
    threshold_expected_profit,
    threshold_f1,
    threshold_youden,
)


def setup_experiment(name: str) -> str:
    """Set MLflow tracking URI to local mlruns/ and activate an experiment."""
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(name)
    return name


# ---------------------------------------------------------------------------
# Threshold strategy dispatcher
# ---------------------------------------------------------------------------

_STRATEGIES = {
    "youden": lambda y_true, y_prob, **kw: threshold_youden(y_true, y_prob),
    "f1":     lambda y_true, y_prob, **kw: threshold_f1(y_true, y_prob),
    "cost_minimization": lambda y_true, y_prob, **kw:
        threshold_cost_minimization(y_true, y_prob, **kw),
    "expected_profit":   lambda y_true, y_prob, **kw:
        threshold_expected_profit(y_true, y_prob, **kw),
}


def _pick_threshold(strategy, y_true, y_prob, kwargs):
    if strategy not in _STRATEGIES:
        raise ValueError(f"Unknown threshold_strategy: {strategy!r}")
    return _STRATEGIES[strategy](y_true, y_prob, **(kwargs or {}))


# ---------------------------------------------------------------------------
# Main logging helper
# ---------------------------------------------------------------------------

def log_training_run(
    model,
    model_name: str,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    params: dict,
    optimal_threshold: float | None = None,
    artifacts: dict[str, str] | None = None,
    threshold_strategy: str = "expected_profit",
    threshold_kwargs: dict | None = None,
    sensitive_map: dict | None = None,
) -> str:
    """Log a complete training run to MLflow.

    Args:
        model: trained sklearn-compatible model with predict_proba
        model_name: display name for the run
        X_test: test features
        y_test: test labels
        params: hyperparameters to log
        optimal_threshold: deprecated — if provided it is used directly and
            threshold_strategy is ignored. Pass None to use strategy dispatch.
        artifacts: optional {name: file_path} to log as artifacts
        threshold_strategy: one of "youden", "f1", "cost_minimization",
            "expected_profit". Default "expected_profit". Ignored when
            optimal_threshold is supplied explicitly.
            Note: "expected_profit" requires loan_amnts, int_rates, terms,
            recoveries arrays passed via threshold_kwargs; if those are absent
            the strategy automatically falls back to "youden".
        threshold_kwargs: extra keyword arguments forwarded to the chosen
            threshold function (e.g. cost_fp, cost_fn for cost_minimization,
            or the four loan arrays for expected_profit).
        sensitive_map: optional dict mapping attribute names to pd.Series or
            array-like column vectors (aligned with X_test/y_test). When
            provided, fairness_report() is called and its per-group metrics
            are logged as MLflow metrics plus a fairness_report.csv artifact.

    Returns:
        MLflow run ID
    """
    with mlflow.start_run(run_name=model_name) as run:
        # Parameters
        mlflow.log_params(params)
        mlflow.log_param("n_features", X_test.shape[1])

        # Resolve threshold
        if optimal_threshold is not None:
            # Legacy path: caller supplies threshold directly
            resolved_strategy = "provided"
        else:
            # Dispatch via strategy
            effective_strategy = threshold_strategy
            effective_kwargs = threshold_kwargs or {}

            if effective_strategy == "expected_profit":
                required_keys = {"loan_amnts", "int_rates", "terms", "recoveries"}
                if not required_keys.issubset(effective_kwargs.keys()):
                    import logging
                    logging.warning(
                        "threshold_strategy='expected_profit' requires %s in "
                        "threshold_kwargs; falling back to 'youden'.",
                        required_keys,
                    )
                    effective_strategy = "youden"
                    effective_kwargs = {}

            optimal_threshold = _pick_threshold(
                effective_strategy, y_test, model.predict_proba(X_test)[:, 1],
                effective_kwargs,
            )
            resolved_strategy = effective_strategy

        mlflow.log_param("threshold_strategy", resolved_strategy)
        mlflow.log_param("optimal_threshold", round(optimal_threshold, 4))

        # Predictions + metrics
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= optimal_threshold).astype(int)
        metrics = compute_metrics(y_test, y_pred, y_prob)
        mlflow.log_metrics(metrics)

        # Bootstrap confidence intervals
        try:
            ci = bootstrap_metrics(y_test, y_pred, y_prob, n_bootstraps=CONFIG.n_bootstraps)
            for metric_name, entry in ci.items():
                mlflow.log_metric(f"{metric_name}_ci_low", entry["ci_low"])
                mlflow.log_metric(f"{metric_name}_ci_high", entry["ci_high"])
        except Exception as exc:
            import logging
            logging.warning("Bootstrap CIs failed: %s", exc)

        # Fairness metrics (optional)
        if sensitive_map is not None:
            try:
                from src.fairness import fairness_report
                from pathlib import Path as _Path
                import pandas as _pd
                fr = fairness_report(
                    model=model,
                    X_test=X_test, y_test=y_test,
                    sensitive_map=sensitive_map,
                    threshold=optimal_threshold,
                    min_group_size=30,
                )
                for _, row in fr.iterrows():
                    if _pd.isna(row.get("demographic_parity_diff")):
                        continue
                    prefix = f"fair_{row['attribute']}_{row['group']}"
                    safe = prefix.replace(" ", "_").replace("/", "_").replace("-", "_")
                    mlflow.log_metric(f"{safe}_demographic_parity_diff", float(row["demographic_parity_diff"]))
                    mlflow.log_metric(f"{safe}_equalized_odds_diff", float(row["equalized_odds_diff"]))
                    mlflow.log_metric(f"{safe}_disparate_impact_ratio", float(row["disparate_impact_ratio"]))
                tmp = _Path("/tmp") / "fairness_report.csv"
                fr.to_csv(tmp, index=False)
                mlflow.log_artifact(str(tmp), artifact_path="fairness")
            except Exception as _exc:
                import logging as _logging
                _logging.warning("fairness_report integration failed: %s", _exc)

        # Model artifact
        mlflow.sklearn.log_model(model, "model")

        # ROC curve artifact
        with tempfile.TemporaryDirectory() as tmpdir:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(fpr, tpr, label=f"AUC={auc(fpr, tpr):.3f}")
            ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title(f"ROC Curve — {model_name}")
            ax.legend()
            roc_path = os.path.join(tmpdir, "roc_curve.png")
            fig.savefig(roc_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            mlflow.log_artifact(roc_path)

        # Additional artifacts
        if artifacts:
            for name, path in artifacts.items():
                if os.path.exists(path):
                    mlflow.log_artifact(path)

        return run.info.run_id
