# tests/test_trainer.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.schema import Session, Players, GameweekStats
from models.prediction.feature_engineer import FeatureEngineer
import xgboost as xgb
import joblib
import os

output_dir = "models/prediction/models"
os.makedirs(output_dir, exist_ok=True)

engineer = FeatureEngineer()
session = Session()

position_map = {
    1: "goalkeeper",
    2: "defender",
    3: "midfielder",
    4: "forward"
}

feature_methods = {
    "goalkeeper": engineer.goalkeeper_features,
    "defender": engineer.defender_features,
    "midfielder": engineer.midfielder_features,
    "forward": engineer.forward_features
}

for pos_code, pos_name in position_map.items():
    players = session.query(Players).filter(Players.position == pos_code).all()
    X, y = [], []

    for player in players:
        stats = session.query(GameweekStats).filter(
            GameweekStats.player_id == player.id,
            GameweekStats.minutes > 0
        ).all()

        for stat in stats:
            features = feature_methods[pos_name](player.id, stat.gameweek_id)
            if features:
                X.append(features)
                y.append(stat.points)

    if X and y:
        model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100)
        model.fit(X, y)
        joblib.dump(model, os.path.join(output_dir, f"{pos_name}_model.pkl"))
        print(f"✅ Trained and saved model for {pos_name}")
    else:
        print(f"⚠️ No training data for {pos_name}")
