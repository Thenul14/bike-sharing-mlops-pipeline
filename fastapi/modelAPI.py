"""REST prediction service. Run from project root using Uvicorn."""
from functools import lru_cache
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import mlflow.pyfunc
import pandas as pd
from src.config import configure_mlflow, MODEL_NAME, MODEL_ALIAS
from src.features import FEATURES

app = FastAPI(title="Bike Sharing Demand API", version="1.0.0")

class BikeInput(BaseModel):
    season: int = Field(ge=1, le=4)
    yr: int = Field(ge=0, le=1)
    mnth: int = Field(ge=1, le=12)
    day: int = Field(ge=1, le=31)
    hr: int = Field(ge=0, le=23)
    holiday: int = Field(ge=0, le=1)
    weekday: int = Field(ge=0, le=6)
    workingday: int = Field(ge=0, le=1)
    weather_situation: int = Field(ge=1, le=4)
    temp: float = Field(ge=-8, le=39)
    atemp: float = Field(ge=-16, le=50)
    hum: float = Field(ge=0, le=100)
    windspeed: float = Field(ge=0, le=67)

@lru_cache(maxsize=1)
def get_model():
    configure_mlflow()
    return mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@{MODEL_ALIAS}")

@app.get("/")
def home():
    return {"message": "Bike Prediction API is running", "docs": "/docs"}

@app.post("/predict")
def predict(data: BikeInput):
    try:
        model = get_model()
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Model unavailable. Run the training DAG and check MLflow.") from exc
    values = data.model_dump()
    df = pd.DataFrame([values], columns=FEATURES)
    prediction = model.predict(df)
    return {"predicted_count": float(prediction[0]), "input": values}
