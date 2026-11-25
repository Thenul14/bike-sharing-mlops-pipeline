import pyarrow as pa
from src.config import redis_client, configure_mlflow
from src.features import split_data
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import mlflow
import mlflow.sklearn
import numpy as np

class LinearRegression:
    def __init__(self):
        self.redis_client = redis_client()

    def train(self):
        configure_mlflow()
        mlflow.set_experiment("linear_regression_experiment")

        # Retrieve data from Redis
        retrieved_data = self.redis_client.get("feature_engineered_data")

        reader = pa.ipc.open_stream(pa.BufferReader(retrieved_data))
        table_restored = reader.read_all()
        df = table_restored.to_pandas()
        print(f"\n{df.info()} \n ")

        X_train, X_test, y_train, y_test = split_data(df)

        # Hyperparameter tuning (Ridge)
        param_grid = {"alpha": [0.1, 1.0, 10.0]}
        grid = GridSearchCV(Ridge(), param_grid, cv=3)
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_

        # Predictions
        y_pred = best_model.predict(X_test)

        # Metrics
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        # Print metrics
        print("Linear Regression results:")
        print(f"R2 Score: {r2}")
        print(f"MAE: {mae}")
        print(f"RMSE: {rmse}")

        # MLflow logging
        with mlflow.start_run(run_name="linear_regression_model") as run:

            mlflow.log_param("model_type", "Ridge")
            mlflow.log_param("alpha", grid.best_params_['alpha'])

            mlflow.log_metric("cv_r2", float(grid.best_score_))
            mlflow.log_metric("r2_score", r2)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("rmse", rmse)

            model_info = mlflow.sklearn.log_model(
                best_model,
                name="model",
                registered_model_name= "BikeShare_Linear"
            )
            self.redis_client.set("linear_model_uri", model_info.model_uri)
            self.redis_client.set("linear_runid", run.info.run_id)


        print("----Model trained, evaluated, and logged to MLflow----\n")
