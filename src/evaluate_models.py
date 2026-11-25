"""Select by cross-validation on training data, not on the held-out test set."""
import mlflow
from mlflow.tracking import MlflowClient
from src.config import configure_mlflow, redis_client, MODEL_NAME, MODEL_ALIAS

class ModelComparer:
    def __init__(self):
        self.redis_client = redis_client()

    def compare_and_select(self):
        configure_mlflow()
        client = MlflowClient()
        candidates = {}
        for key in ("linear", "rf", "gb"):
            run_id = self.redis_client.get(f"{key}_runid")
            model_uri = self.redis_client.get(f"{key}_model_uri")
            if not run_id or not model_uri:
                raise RuntimeError(f"Missing training result for {key}; run all training tasks first.")
            metrics = client.get_run(run_id.decode()).data.metrics
            candidates[key] = (model_uri.decode(), metrics)

        best_name = max(candidates, key=lambda name: candidates[name][1]["cv_r2"])
        model_uri, metrics = candidates[best_name]
        mlflow.set_experiment("model_selection_experiment")
        with mlflow.start_run(run_name="best_model_selection"):
            mlflow.log_param("best_model", best_name)
            for name, (_, scores) in candidates.items():
                mlflow.log_metric(f"{name}_cv_r2", scores["cv_r2"])
            mlflow.log_metrics({"best_cv_r2": metrics["cv_r2"],
                                "selected_test_r2": metrics["r2_score"],
                                "selected_test_mae": metrics["mae"],
                                "selected_test_rmse": metrics["rmse"]})
            version = mlflow.register_model(model_uri, MODEL_NAME)
            client.set_registered_model_alias(MODEL_NAME, MODEL_ALIAS, version.version)
            mlflow.log_param("registered_version", version.version)
        print(f"Selected {best_name} by training CV; held-out R2={metrics['r2_score']:.4f}")
