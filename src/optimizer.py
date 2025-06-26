# src/optimizer.py

import pandas as pd
import os
from pulp import LpProblem, LpVariable, LpMaximize, lpSum, LpBinary, PULP_CBC_CMD

DATA_PATH = "data/processed/predictions.csv"
OUTPUT_PATH = "data/processed/optimized_squad.csv"

def load_data():
    df = pd.read_csv(DATA_PATH)
    df["cost"] = df["now_cost"] / 10  # convert to £
    return df

def optimize_team(df):
    # Create optimization problem
    prob = LpProblem("FPL_Lineup_Optimizer", LpMaximize)

    # Create binary decision variables for each player
    players = df.index.tolist()
    x = LpVariable.dicts("select", players, cat=LpBinary)

    # Objective: Maximize predicted fantasy points
    prob += lpSum(df.loc[i, "predicted_points"] * x[i] for i in players), "Total_Points"

    # Constraint: Exactly 15 players
    prob += lpSum(x[i] for i in players) == 15, "Total_Players"

    # Constraint: Budget limit
    prob += lpSum(df.loc[i, "cost"] * x[i] for i in players) <= 100, "Total_Budget"

    # Constraint: Position requirements
    for pos, count in [("Goalkeeper", 2), ("Defender", 5), ("Midfielder", 5), ("Forward", 3)]:
        prob += lpSum(x[i] for i in players if df.loc[i, "position"] == pos) == count, f"{pos}_Count"

    # Constraint: Max 3 players per club
    for team in df["team_name"].unique():
        prob += lpSum(x[i] for i in players if df.loc[i, "team_name"] == team) <= 3, f"{team}_Max_3"

    # Solve
    prob.solve(PULP_CBC_CMD(msg=1))

    # Output selected players
    selected_players = df[[x[i].varValue == 1 for i in players]].copy()
    selected_players.to_csv(OUTPUT_PATH, index=False)
    print("✅ Optimized squad saved to", OUTPUT_PATH)

if __name__ == "__main__":
    df = load_data()
    optimize_team(df)
