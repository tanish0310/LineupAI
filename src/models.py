# src/models.py

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

PROCESSED_DIR = "data/processed"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def load_data():
    df = pd.read_csv(f"{PROCESSED_DIR}/player_features.csv")
    df = df[df["minutes"] > 90]  # Filter out low-playing players

    features = [
        "form", "points_per_game", "now_cost", "value_per_million",
        "selected_by_percent", "minutes"
    ]
    target = "total_points"

    X = df[features].copy()
    y = df[target]

    return df, X, y

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = GradientBoostingRegressor()
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    from math import sqrt
    rmse = sqrt(mean_squared_error(y_test, y_pred))

    print(f"✅ Model trained | RMSE: {rmse:.2f}")

    return model, scaler

def save_model(model, scaler):
    joblib.dump(model, f"{MODEL_DIR}/points_predictor.pkl")
    joblib.dump(scaler, f"{MODEL_DIR}/model_scaler.pkl")
    print("✅ Model and scaler saved!")

def generate_predictions(df, model, scaler):
    features = [
        "form", "points_per_game", "now_cost", "value_per_million",
        "selected_by_percent", "minutes"
    ]
    X_scaled = scaler.transform(df[features])
    df["predicted_points"] = model.predict(X_scaled)

    df.to_csv(f"{PROCESSED_DIR}/predictions.csv", index=False)
    print("✅ Predictions saved to data/processed/predictions.csv")

if __name__ == "__main__":
    df, X, y = load_data()
    model, scaler = train_model(X, y)
    save_model(model, scaler)
    generate_predictions(df, model, scaler)
