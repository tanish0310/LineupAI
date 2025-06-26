# src/data_collection.py

import requests
import pandas as pd
import os

DATA_DIR = "data/raw"
os.makedirs(DATA_DIR, exist_ok=True)

def fetch_bootstrap():
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    response = requests.get(url)
    data = response.json()

    pd.DataFrame(data['elements']).to_csv(f"{DATA_DIR}/players.csv", index=False)
    pd.DataFrame(data['teams']).to_csv(f"{DATA_DIR}/teams.csv", index=False)
    pd.DataFrame(data['events']).to_csv(f"{DATA_DIR}/gameweeks.csv", index=False)

    print("✅ bootstrap data fetched")

def fetch_fixtures():
    url = "https://fantasy.premierleague.com/api/fixtures/"
    response = requests.get(url)
    fixtures = response.json()
    pd.DataFrame(fixtures).to_csv(f"{DATA_DIR}/fixtures.csv", index=False)
    print("✅ fixtures data fetched")

def update_all_data():
    fetch_bootstrap()
    fetch_fixtures()
    print("✅ All raw data updated!")

if __name__ == "__main__":
    update_all_data()
