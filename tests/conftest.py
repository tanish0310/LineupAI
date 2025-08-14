# tests/conftest.py
import pytest
import json

@pytest.fixture
def sample_squad():
    """Sample 15-player squad for testing."""
    return [
        {"id": i, "name": f"Player {i}", "position": (i % 4) + 1, 
         "team": (i % 20) + 1, "price": 4.0 + (i % 12) * 0.5, 
         "selling_price": 4.0 + (i % 12) * 0.5}
        for i in range(1, 16)
    ]

@pytest.fixture
def sample_predictions():
    """Sample prediction data for testing."""
    return {
        str(i): {
            "points": 4.0 + (i % 8) * 1.5,
            "confidence": 0.7 + (i % 3) * 0.1
        }
        for i in range(1, 16)
    }

@pytest.fixture
def valid_squad_request():
    """Valid squad optimization request."""
    return {
        "gameweek_id": 1,
        "budget": 1000,
        "strategy": "balanced"
    }

@pytest.fixture
def valid_transfer_request(sample_squad):
    """Valid transfer analysis request."""
    return {
        "current_squad": sample_squad,
        "gameweek_id": 1,
        "free_transfers": 1,
        "budget_remaining": 0.0,
        "analyze_hits": True,
        "multi_gw_horizon": 3
    }

@pytest.fixture
def mock_player_data():
    """Mock player data for testing."""
    return {
        "id": 1,
        "name": "Test Player",
        "position": 3,  # Midfielder
        "team": 1,
        "price": 8.0,
        "form": 5.2,
        "ownership": 25.4
    }
