"""Shared environment and project configuration."""
import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")
DATA_PATH = BASE_DIR / "bike_share" / "hour.csv"
MODEL_NAME = "BikeShare_Best_Model"
MODEL_ALIAS = "champion"

def database_url():
    url = os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError("Set DATABASE_URL in .env before running database tasks.")
    return url

def redis_client():
    import redis
    return redis.Redis.from_url(os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0"))

def configure_mlflow():
    import mlflow
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
