import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # Database Configuration
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost/fpl_optimizer")
    
    # FPL API Configuration
    FPL_BASE_URL = "https://fantasy.premierleague.com/api/"
    FPL_BOOTSTRAP_URL = f"{FPL_BASE_URL}bootstrap-static/"
    FPL_FIXTURES_URL = f"{FPL_BASE_URL}fixtures/"
    FPL_LIVE_URL = f"{FPL_BASE_URL}event/{}/live/"
    FPL_PLAYER_HISTORY_URL = f"{FPL_BASE_URL}element-summary/{}/"
    
    # Data Collection Settings
    REQUEST_TIMEOUT = 30
    REQUEST_RETRY_COUNT = 3
    REQUEST_DELAY = 1  # seconds between requests
    
    # Model Settings
    MODEL_CACHE_HOURS = 24
    PREDICTION_CONFIDENCE_THRESHOLD = 0.7
    
    # Budget and Squad Constraints
    TOTAL_BUDGET = 1000  # £100.0m in tenths
    SQUAD_SIZE = 15
    STARTING_XI_SIZE = 11
    MAX_PLAYERS_PER_TEAM = 3
    
    # Position Constraints
    POSITION_LIMITS = {
        1: {'min': 2, 'max': 2, 'name': 'Goalkeeper'},  # GK
        2: {'min': 5, 'max': 5, 'name': 'Defender'},    # DEF
        3: {'min': 5, 'max': 5, 'name': 'Midfielder'},  # MID
        4: {'min': 3, 'max': 3, 'name': 'Forward'}      # FWD
    }

settings = Settings()
