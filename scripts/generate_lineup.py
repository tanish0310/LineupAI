# scripts/generate_lineup.py

import pandas as pd

INPUT_PATH = "data/processed/optimized_squad.csv"
OUTPUT_PATH = "data/processed/optimized_cleaned.csv"

def clean_output():
    df = pd.read_csv(INPUT_PATH)

    keep_cols = [
        "web_name", "team_name", "position", "now_cost",
        "predicted_points", "total_points", "form",
        "points_per_game", "minutes"
    ]

    df_cleaned = df[keep_cols].sort_values("predicted_points", ascending=False)
    df_cleaned["now_cost"] = df_cleaned["now_cost"] / 10  # convert to £

    df_cleaned.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Cleaned squad saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    clean_output()
