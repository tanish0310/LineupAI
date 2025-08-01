import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.optimization.squad_optimizer import SquadOptimizer
from models.prediction.player_predictor import PlayerPredictor

predictor = PlayerPredictor()
predictions = predictor.predict_gameweek_points(gameweek_id=1)

optimizer = SquadOptimizer()
optimal_squad = optimizer.build_optimal_squad(predictions)
starting_xi = optimizer.optimize_starting_xi(optimal_squad, predictions)
captain_data = optimizer.recommend_captain_vice(starting_xi, predictions)

print("✅ Optimal 15-man squad:", optimal_squad)
print("🔵 Starting XI:", starting_xi)
print("🧢 Captain:", captain_data['captain'])
print("🧢 Vice Captain:", captain_data['vice_captain'])
print("👑 Top 3 captain options:", captain_data['top_3'])
