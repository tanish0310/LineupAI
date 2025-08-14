# test_api_complete.py
import pytest
import requests
import json
import asyncio
from datetime import datetime
import time
from typing import Dict, List
import os

# Test configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
TEST_API_KEY = os.getenv("TEST_API_KEY", "test-api-key-123")

# Test headers for authentication
TEST_HEADERS = {
    "Authorization": f"Bearer {TEST_API_KEY}",
    "Content-Type": "application/json"
}

class TestFPLOptimizerAPI:
    """Comprehensive test suite for FPL Optimizer Pro API."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment."""
        # Wait for API to be ready
        max_retries = 30
        for i in range(max_retries):
            try:
                response = requests.get(f"{API_BASE_URL}/health")
                if response.status_code == 200:
                    break
            except requests.exceptions.ConnectionError:
                if i == max_retries - 1:
                    pytest.fail("API not available after maximum retries")
                time.sleep(1)
    
    # Health and System Tests
    def test_health_check(self):
        """Test comprehensive health check endpoint."""
        response = requests.get(f"{API_BASE_URL}/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert data["status"] in ["healthy", "degraded"]
        assert "components" in data
        assert "database" in data["components"]
        assert "redis" in data["components"]
        assert "ml_models" in data["components"]
        assert "timestamp" in data
        assert "version" in data
    
    def test_metrics_endpoint(self):
        """Test Prometheus metrics endpoint."""
        response = requests.get(f"{API_BASE_URL}/metrics")
        assert response.status_code == 200
        
        # Check for basic Prometheus metrics format
        content = response.text
        assert "api_requests_total" in content
        assert "api_request_duration_seconds" in content
    
    def test_authentication_required(self):
        """Test that protected endpoints require authentication."""
        # Test without auth header
        response = requests.post(f"{API_BASE_URL}/predictions/generate")
        assert response.status_code == 401
        
        # Test with invalid auth header
        invalid_headers = {"Authorization": "Bearer invalid-key"}
        response = requests.post(
            f"{API_BASE_URL}/predictions/generate",
            headers=invalid_headers
        )
        assert response.status_code == 401
    
    # Prediction Tests
    def test_generate_predictions(self):
        """Test prediction generation endpoint."""
        request_data = {"gameweek_id": 1}
        
        response = requests.post(
            f"{API_BASE_URL}/predictions/generate",
            json=request_data,
            headers=TEST_HEADERS
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert "predictions" in data
        assert "gameweek_id" in data
        assert "generated_at" in data
        assert "total_players" in data
        assert data["gameweek_id"] == 1
        
        # Verify prediction format
        if data["predictions"]:
            prediction = data["predictions"][0]
            required_fields = [
                'player_id', 'name', 'team', 'position', 
                'price', 'predicted_points', 'confidence'
            ]
            for field in required_fields:
                assert field in prediction
            
            # Verify data types
            assert isinstance(prediction['player_id'], int)
            assert isinstance(prediction['predicted_points'], (int, float))
            assert isinstance(prediction['confidence'], (int, float))
            assert 0 <= prediction['confidence'] <= 1
    
    def test_predictions_missing_gameweek(self):
        """Test prediction generation with missing gameweek."""
        response = requests.post(
            f"{API_BASE_URL}/predictions/generate",
            json={},
            headers=TEST_HEADERS
        )
        
        assert response.status_code == 400
        data = response.json()
        assert "gameweek_id required" in data["detail"]
    
    # Squad Optimization Tests
    def test_squad_optimization(self):
        """Test complete squad optimization."""
        request_data = {
            "gameweek_id": 1,
            "budget": 1000,  # £100.0m
            "strategy": "balanced"
        }
        
        response = requests.post(
            f"{API_BASE_URL}/squad/optimize",
            json=request_data,
            headers=TEST_HEADERS
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        required_fields = [
            'starting_xi', 'bench', 'captain', 'vice_captain',
            'formation', 'total_cost', 'predicted_points',
            'budget_remaining', 'squad_analysis', 'validation',
            'captain_options'
        ]
        for field in required_fields:
            assert field in data, f"Missing field: {field}"
        
        # Verify starting XI structure
        starting_xi = data['starting_xi']
        assert 'GK' in starting_xi
        assert 'DEF' in starting_xi
        assert 'MID' in starting_xi
        assert 'FWD' in starting_xi
        
        # Verify position constraints
        assert len(starting_xi['GK']) == 1
        assert 3 <= len(starting_xi['DEF']) <= 5
        assert 2 <= len(starting_xi['MID']) <= 5
        assert 1 <= len(starting_xi['FWD']) <= 3
        
        # Verify total squad size
        total_starters = sum(len(pos) for pos in starting_xi.values())
        assert total_starters == 11
        assert len(data['bench']) == 4
        
        # Verify budget constraint
        assert data['total_cost'] <= 1000
        assert data['budget_remaining'] >= 0
        
        # Verify captain and vice-captain
        captain = data['captain']
        vice_captain = data['vice_captain']
        assert captain['id'] != vice_captain['id']
        
        # Verify formation is valid
        valid_formations = [
            "3-4-3", "3-5-2", "4-3-3", "4-4-2", 
            "4-5-1", "5-3-2", "5-4-1"
        ]
        assert data['formation'] in valid_formations
        
        # Verify captain options
        assert len(data['captain_options']) >= 1
        for option in data['captain_options']:
            assert 'rank' in option
            assert 'player' in option
            assert 'expected_points' in option
    
    def test_squad_optimization_with_formation(self):
        """Test squad optimization with specific formation."""
        request_data = {
            "gameweek_id": 1,
            "budget": 1000,
            "formation": "3-5-2"
        }
        
        response = requests.post(
            f"{API_BASE_URL}/squad/optimize",
            json=request_data,
            headers=TEST_HEADERS
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify requested formation is used
        starting_xi = data['starting_xi']
        assert len(starting_xi['DEF']) == 3
        assert len(starting_xi['MID']) == 5
        assert len(starting_xi['FWD']) == 2
    
    def test_squad_optimization_with_locked_players(self):
        """Test squad optimization with locked players."""
        request_data = {
            "gameweek_id": 1,
            "budget": 1000,
            "locked_players": [1, 2, 3]  # Lock specific player IDs
        }
        
        response = requests.post(
            f"{API_BASE_URL}/squad/optimize",
            json=request_data,
            headers=TEST_HEADERS
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify locked players are included (would need actual player data to verify)
        assert "starting_xi" in data
        assert "bench" in data
    
    # Squad Analysis Tests
    def test_squad_analysis(self):
        """Test current squad analysis."""
        request_data = {
            "player_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            "gameweek_id": 1,
            "free_transfers": 1,
            "budget_remaining": 0.5
        }
        
        response = requests.post(
            f"{API_BASE_URL}/squad/analyze",
            json=request_data,
            headers=TEST_HEADERS
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify analysis structure
        required_fields = [
            'current_points', 'optimal_points', 'improvement_potential',
            'squad_rank_percentile', 'injury_concerns', 'price_change_alerts',
            'analysis_timestamp'
        ]
        for field in required_fields:
            assert field in data, f"Missing field: {field}"
        
        # Verify data types
        assert isinstance(data['current_points'], (int, float))
        assert isinstance(data['optimal_points'], (int, float))
        assert isinstance(data['improvement_potential'], (int, float))
        assert isinstance(data['squad_rank_percentile'], int)
        assert isinstance(data['injury_concerns'], list)
        assert isinstance(data['price_change_alerts'], list)
    
    # Transfer Analysis Tests
    def test_transfer_analysis(self):
        """Test transfer analysis with recommendations."""
        request_data = {
            "current_squad": [{"id": i, "name": f"Player {i}", "position": (i % 4) + 1, 
                             "team": (i % 20) + 1, "price": 5.0, "selling_price": 5.0} 
                            for i in range(1, 16)],
            "gameweek_id": 1,
            "free_transfers": 1,
            "budget_remaining": 0.0,
            "analyze_hits": True,
            "multi_gw_horizon": 3
        }
        
        response = requests.post(
            f"{API_BASE_URL}/transfers/analyze",
            json=request_data,
            headers=TEST_HEADERS
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        required_fields = [
            'current_squad_analysis', 'transfer_recommendations',
            'injury_concerns', 'price_change_alerts'
        ]
        for field in required_fields:
            assert field in data, f"Missing field: {field}"
        
        # Verify transfer recommendations structure
        if data['transfer_recommendations']:
            rec = data['transfer_recommendations'][0]
            rec_fields = [
                'transfers', 'total_improvement', 'total_cost',
                'transfers_used', 'hit_required', 'hit_cost',
                'net_improvement', 'recommendation'
            ]
            for field in rec_fields:
                assert field in rec, f"Missing recommendation field: {field}"
            
            # Verify recommendation values
            assert rec['recommendation'] in ["RECOMMENDED", "MARGINAL", "AVOID"]
            
            # Verify transfer details
            if rec['transfers']:
                transfer = rec['transfers'][0]
                transfer_fields = [
                    'player_out', 'player_in', 'point_improvement',
                    'cost', 'reasoning', 'confidence'
                ]
                for field in transfer_fields:
                    assert field in transfer, f"Missing transfer field: {field}"
                
                # Verify player structures
                for player_key in ['player_out', 'player_in']:
                    player = transfer[player_key]
                    player_fields = ['id', 'name', 'position', 'team', 'price']
                    for field in player_fields:
                        assert field in player, f"Missing player field: {field}"
    
    def test_transfer_analysis_missing_data(self):
        """Test transfer analysis with missing required data."""
        # Test missing current_squad
        response = requests.post(
            f"{API_BASE_URL}/transfers/analyze",
            json={"gameweek_id": 1},
            headers=TEST_HEADERS
        )
        assert response.status_code == 400
        assert "current_squad and gameweek_id required" in response.json()["detail"]
        
        # Test missing gameweek_id
        response = requests.post(
            f"{API_BASE_URL}/transfers/analyze",
            json={"current_squad": []},
            headers=TEST_HEADERS
        )
        assert response.status_code == 400
    
    # Transfer Hit Analysis Tests
    def test_transfer_hit_analysis(self):
        """Test transfer hit analysis."""
        request_data = {
            "transfer_options": [
                {
                    "players_out": [{"id": 1, "price": 8.0}],
                    "players_in": [{"id": 2, "price": 9.0}],
                    "improvement": 3.5,
                    "transfers_used": 2
                }
            ],
            "gameweek_id": 1,
            "analysis_horizon": 4
        }
        
        response = requests.post(
            f"{API_BASE_URL}/transfers/hit-analysis",
            json=request_data,
            headers=TEST_HEADERS
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert "hit_analyses" in data
        assert "scenario_comparison" in data
        assert "analysis_parameters" in data
        
        # Verify analysis parameters
        params = data["analysis_parameters"]
        assert params["gameweek_start"] == 1
        assert params["analysis_horizon"] == 4
        assert params["scenarios_analyzed"] == 1
    
    # Multi-Gameweek Planning Tests
    def test_multi_gameweek_planning(self):
        """Test multi-gameweek strategic planning."""
        request_data = {
            "current_squad": [{"id": i, "name": f"Player {i}", "position": (i % 4) + 1} 
                            for i in range(1, 16)],
            "current_gameweek": 1,
            "planning_horizon": 6,
            "available_chips": ["wildcard", "triple_captain"],
            "strategy_type": "aggressive"
        }
        
        response = requests.post(
            f"{API_BASE_URL}/planning/multi-gameweek",
            json=request_data,
            headers=TEST_HEADERS
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert "strategic_plan" in data
        assert "plan_summary" in data
        
        # Verify plan summary
        summary = data["plan_summary"]
        assert summary["planning_horizon"] == 6
        assert summary["strategy_type"] == "aggressive"
        assert summary["chips_available"] == 2
    
    # Player Analysis Tests
    def test_player_analysis(self):
        """Test individual player analysis."""
        player_id = 1
        gameweek_id = 1
        
        response = requests.get(
            f"{API_BASE_URL}/player/{player_id}/analysis?gameweek_id={gameweek_id}",
            headers=TEST_HEADERS
        )
        
        # Note: This might return 404 if player doesn't exist in test data
        if response.status_code == 200:
            data = response.json()
            
            required_fields = [
                'player_data', 'prediction', 'captain_analysis',
                'upcoming_fixtures', 'recent_performance',
                'recommendation', 'reasoning'
            ]
            for field in required_fields:
                assert field in data, f"Missing field: {field}"
            
            # Verify recommendation values
            assert data['recommendation'] in ["STRONG BUY", "BUY", "HOLD", "SELL"]
            assert isinstance(data['reasoning'], list)
        elif response.status_code == 404:
            # Player not found - acceptable for test environment
            assert "Player not found" in response.json()["detail"]
        else:
            pytest.fail(f"Unexpected status code: {response.status_code}")
    
    # Utility Endpoint Tests
    def test_get_current_gameweek(self):
        """Test current gameweek endpoint."""
        response = requests.get(f"{API_BASE_URL}/gameweek/current")
        assert response.status_code == 200
        
        data = response.json()
        assert "current_gameweek" in data
        assert "timestamp" in data
        assert isinstance(data["current_gameweek"], int)
        assert 1 <= data["current_gameweek"] <= 38
    
    def test_get_teams(self):
        """Test teams endpoint."""
        response = requests.get(f"{API_BASE_URL}/teams")
        assert response.status_code == 200
        
        data = response.json()
        assert "teams" in data
        assert "total_teams" in data
        assert "timestamp" in data
        assert isinstance(data["teams"], list)
        assert data["total_teams"] == len(data["teams"])
    
    def test_trigger_data_update(self):
        """Test data update trigger."""
        response = requests.post(
            f"{API_BASE_URL}/data/update",
            headers=TEST_HEADERS
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "Data update initiated"
        assert "timestamp" in data
    
    def test_model_performance(self):
        """Test model performance metrics."""
        response = requests.get(f"{API_BASE_URL}/model/performance")
        assert response.status_code == 200
        
        data = response.json()
        assert "model_performance" in data
        assert "timestamp" in data

# Performance and Load Tests
class TestAPIPerformance:
    """Performance testing for critical endpoints."""
    
    def test_prediction_generation_performance(self):
        """Test prediction generation response time."""
        request_data = {"gameweek_id": 1}
        
        start_time = time.time()
        response = requests.post(
            f"{API_BASE_URL}/predictions/generate",
            json=request_data,
            headers=TEST_HEADERS
        )
        end_time = time.time()
        
        assert response.status_code == 200
        response_time = end_time - start_time
        
        # Should respond within 30 seconds for first request (cache miss)
        assert response_time < 30, f"Prediction generation took {response_time:.2f}s"
        
        # Test cached response (should be much faster)
        start_time = time.time()
        response = requests.post(
            f"{API_BASE_URL}/predictions/generate",
            json=request_data,
            headers=TEST_HEADERS
        )
        end_time = time.time()
        
        cached_response_time = end_time - start_time
        assert cached_response_time < 2, f"Cached prediction took {cached_response_time:.2f}s"
    
    def test_squad_optimization_performance(self):
        """Test squad optimization response time."""
        request_data = {
            "gameweek_id": 1,
            "budget": 1000,
            "strategy": "balanced"
        }
        
        start_time = time.time()
        response = requests.post(
            f"{API_BASE_URL}/squad/optimize",
            json=request_data,
            headers=TEST_HEADERS
        )
        end_time = time.time()
        
        assert response.status_code == 200
        response_time = end_time - start_time
        
        # Should complete within 60 seconds
        assert response_time < 60, f"Squad optimization took {response_time:.2f}s"
    
    def test_concurrent_requests(self):
        """Test handling of concurrent requests."""
        import concurrent.futures
        
        def make_prediction_request():
            return requests.post(
                f"{API_BASE_URL}/predictions/generate",
                json={"gameweek_id": 1},
                headers=TEST_HEADERS
            )
        
        # Test 5 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_prediction_request) for _ in range(5)]
            responses = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All requests should succeed
        for response in responses:
            assert response.status_code == 200

# Integration Tests
class TestAPIIntegration:
    """End-to-end integration tests."""
    
    def test_complete_fpl_workflow(self):
        """Test complete FPL optimization workflow."""
        # Step 1: Get current gameweek
        gw_response = requests.get(f"{API_BASE_URL}/gameweek/current")
        assert gw_response.status_code == 200
        current_gw = gw_response.json()["current_gameweek"]
        
        # Step 2: Generate predictions
        pred_response = requests.post(
            f"{API_BASE_URL}/predictions/generate",
            json={"gameweek_id": current_gw},
            headers=TEST_HEADERS
        )
        assert pred_response.status_code == 200
        predictions = pred_response.json()["predictions"]
        assert len(predictions) > 0
        
        # Step 3: Build optimal squad
        squad_response = requests.post(
            f"{API_BASE_URL}/squad/optimize",
            json={"gameweek_id": current_gw, "budget": 1000},
            headers=TEST_HEADERS
        )
        assert squad_response.status_code == 200
        optimal_squad = squad_response.json()
        
        # Step 4: Analyze squad
        squad_players = []
        for pos_players in optimal_squad["starting_xi"].values():
            squad_players.extend([p["id"] for p in pos_players])
        squad_players.extend([p["id"] for p in optimal_squad["bench"]])
        
        analysis_response = requests.post(
            f"{API_BASE_URL}/squad/analyze",
            json={
                "player_ids": squad_players,
                "gameweek_id": current_gw,
                "free_transfers": 1
            },
            headers=TEST_HEADERS
        )
        assert analysis_response.status_code == 200
        
        # Verify workflow completed successfully
        analysis = analysis_response.json()
        assert "current_points" in analysis
        assert "improvement_potential" in analysis

if __name__ == "__main__":
    # Run tests with pytest
    import sys
    
    # Basic smoke tests that can run quickly
    def run_smoke_tests():
        """Run quick smoke tests to verify API is working."""
        tester = TestFPLOptimizerAPI()
        tester.setup()
        
        try:
            print("🔍 Running API smoke tests...")
            
            tester.test_health_check()
            print("✅ Health check passed")
            
            tester.test_get_current_gameweek()
            print("✅ Current gameweek test passed")
            
            # Skip auth test in development
            if os.getenv("ENVIRONMENT") != "development":
                tester.test_authentication_required()
                print("✅ Authentication test passed")
            
            print("\n🎉 All smoke tests passed! API is working correctly.")
            
        except Exception as e:
            print(f"❌ Smoke test failed: {str(e)}")
            sys.exit(1)
    
    # Run comprehensive tests
    def run_comprehensive_tests():
        """Run full test suite."""
        pytest_args = [
            __file__,
            "-v",  # Verbose output
            "--tb=short",  # Short traceback format
            "--durations=10",  # Show 10 slowest tests
        ]
        
        # Add markers for different test types
        if "--smoke" in sys.argv:
            pytest_args.extend(["-m", "not slow"])
        elif "--performance" in sys.argv:
            pytest_args.extend(["-k", "performance"])
        elif "--integration" in sys.argv:
            pytest_args.extend(["-k", "integration"])
        
        exit_code = pytest.main(pytest_args)
        sys.exit(exit_code)
    
    if "--smoke" in sys.argv:
        run_smoke_tests()
    else:
        run_comprehensive_tests()
