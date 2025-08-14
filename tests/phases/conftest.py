# tests/phases/conftest.py
import pytest

@pytest.fixture
def phase1_data():
    """Test data specific to Phase 1 (Data Collection)."""
    return {
        "test_players": list(range(1, 21)),
        "test_gameweek": 1,
        "expected_data_fields": ["id", "name", "position", "team", "price"]
    }

@pytest.fixture
def phase2_data():
    """Test data specific to Phase 2 (ML Models)."""
    return {
        "test_features": ["form", "fixtures", "price", "ownership"],
        "expected_accuracy": 0.75,
        "test_predictions_count": 100
    }

@pytest.fixture
def phase3_data():
    """Test data specific to Phase 3 (Optimization)."""
    return {
        "budget_constraint": 1000,
        "position_constraints": {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3},
        "team_constraint": 3
    }

@pytest.fixture
def phase4_data():
    """Test data specific to Phase 4 (Transfers)."""
    return {
        "free_transfers": 1,
        "hit_cost": 4,
        "max_transfers": 15
    }
