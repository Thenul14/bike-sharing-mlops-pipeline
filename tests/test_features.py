import pandas as pd
from src.features import FEATURES, model_features, split_data
from src.config import DATA_PATH

def test_feature_contract():
    df = pd.read_csv(DATA_PATH)
    df.insert(5, "day", pd.to_datetime(df["dteday"]).dt.day)
    df = df.rename(columns={"weathersit": "weather_situation"})
    X = model_features(df)
    assert list(X.columns) == FEATURES
    assert not X.isna().any().any()
    assert "cnt" not in X and "registered" not in X and "casual" not in X
    assert X.iloc[0]["hum"] == df.iloc[0]["hum"] * 100
    a, b, c, d = split_data(df)
    assert len(a) + len(b) == len(df)
    assert set(a.index).isdisjoint(b.index)
    assert a.index.equals(c.index) and b.index.equals(d.index)
    assert list(a.columns) == FEATURES
