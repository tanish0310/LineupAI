import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.prediction.player_predictor import PlayerPredictor

predictor = PlayerPredictor()

# Train models
predictor.train_position_models()

# Predict
results = predictor.predict_gameweek_points(gameweek_id=39)  
for pid, info in results.items():
    print(f"🔹 {info['name']} ({info['position']}): {info['points']:.2f} pts (confidence: {info['confidence']})")

