import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.prediction.feature_engineer import FeatureEngineer


fe = FeatureEngineer()

player_id = 1
gameweek_id = 3  # Pick any gameweek after the player has stats

print("✅ GK Features:", fe.goalkeeper_features(player_id, gameweek_id))
print("✅ DEF Features:", fe.defender_features(player_id, gameweek_id))
print("✅ MID Features:", fe.midfielder_features(player_id, gameweek_id))
print("✅ FWD Features:", fe.forward_features(player_id, gameweek_id))

