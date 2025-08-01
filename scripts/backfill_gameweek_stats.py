import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.collection.fpl_collector import FPLDataCollector
from time import sleep

if __name__ == "__main__":
    collector = FPLDataCollector()
    
    # Optional: Refresh core data
    collector.fetch_bootstrap_data()
    collector.fetch_fixtures()

    # Fetch historical gameweek stats from GW 1 to 38
    for gw in range(1, 39):
        print(f"📦 Fetching Gameweek {gw} data...")
        collector.fetch_gameweek_live_data(gw)
        sleep(1)  # Avoid hitting the API too fast

    print("✅ All gameweek stats collected.")
