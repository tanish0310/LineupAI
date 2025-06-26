# src/data_processing.py

import pandas as pd
import os

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

def load_data():
    players = pd.read_csv(f"{RAW_DIR}/players.csv")
    teams = pd.read_csv(f"{RAW_DIR}/teams.csv")
    fixtures = pd.read_csv(f"{RAW_DIR}/fixtures.csv")
    return players, teams, fixtures

def clean_and_engineer(players, teams, fixtures):
    # Add readable team and position names
    team_map = teams.set_index("id")["name"]
    players["team_name"] = players["team"].map(team_map)

    position_map = {
        1: "Goalkeeper",
        2: "Defender",
        3: "Midfielder",
        4: "Forward"
    }
    players["position"] = players["element_type"].map(position_map)

    # Feature engineering
    players["value_per_million"] = players["total_points"] / (players["now_cost"] / 10)
    players["points_per_game"] = pd.to_numeric(players["points_per_game"], errors='coerce').fillna(0.0)
    players["selected_by_percent"] = pd.to_numeric(players["selected_by_percent"], errors='coerce').fillna(0.0)
    players["form"] = pd.to_numeric(players["form"], errors='coerce').fillna(0.0)

    # Map next opponent team and difficulty
    fixtures = fixtures[fixtures["event"].notna()]  # only include upcoming fixtures
    upcoming = fixtures.groupby("team_h")["team_a"].first().reset_index().rename(columns={
        "team_h": "team_id", "team_a": "opponent_team_id"
    })
    players = players.merge(upcoming, left_on="team", right_on="team_id", how="left")

    opponent_map = team_map.to_dict()
    players["opponent_team"] = players["opponent_team_id"].map(opponent_map)

    return players

def save_processed_data(players):
    players.to_csv(f"{PROCESSED_DIR}/player_features.csv", index=False)
    print("✅ Processed player data saved to data/processed/player_features.csv")

if __name__ == "__main__":
    players, teams, fixtures = load_data()
    processed_players = clean_and_engineer(players, teams, fixtures)
    save_processed_data(processed_players)
