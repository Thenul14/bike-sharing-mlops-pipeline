"""Consistent training and serving feature contract."""
from sklearn.model_selection import train_test_split

FEATURES = ["season", "yr", "mnth", "day", "hr", "holiday", "weekday",
            "workingday", "weather_situation", "temp", "atemp", "hum", "windspeed"]

def model_features(df):
    # Preserve the original tree models' conversion to physical units.
    X = df.loc[:, FEATURES].copy()
    X["temp"] = X["temp"] * 47 - 8
    X["atemp"] = X["atemp"] * 66 - 16
    X["hum"] = X["hum"] * 100
    X["windspeed"] = X["windspeed"] * 67
    return X

def split_data(df):
    return train_test_split(model_features(df), df["cnt"], test_size=0.2, random_state=42)
