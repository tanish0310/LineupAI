# conftest.py (Project root)
import pytest
import requests
import time
import os
from typing import Dict, List

@pytest.fixture(scope="session")
def api_base_url():
    """Base URL for API testing."""
    return os.getenv("API_BASE_URL", "http://localhost:8000")

@pytest.fixture(scope="session")
def auth_headers():
    """Authentication headers for API requests."""
    api_key = os.getenv("TEST_API_KEY", "test-api-key-123")
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

@pytest.fixture(scope="session", autouse=True)
def ensure_api_ready(api_base_url):
    """Ensure API is ready before running tests."""
    max_retries = 30
    for i in range(max_retries):
        try:
            response = requests.get(f"{api_base_url}/health", timeout=5)
            if response.status_code == 200:
                print(f"\n✅ API is ready at {api_base_url}")
                return
        except requests.exceptions.RequestException:
            pass
        
        if i == max_retries - 1:
            pytest.exit("❌ API not available after maximum retries")
        
        print(f"⏳ Waiting for API... ({i+1}/{max_retries})")
        time.sleep(1)

@pytest.fixture
def sample_gameweek():
    """Sample gameweek ID for testing."""
    return 1

@pytest.fixture
def sample_budget():
    """Sample budget for testing (£100m in tenths)."""
    return 1000

