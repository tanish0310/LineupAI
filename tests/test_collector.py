import pytest
import requests_mock
from datetime import datetime
from data.collection.fpl_collector import FPLDataCollector
from data.storage.database import db_manager, Player, Team, Gameweek

class TestFPLDataCollector:
    def setup_method(self):
        """Setup test database and collector"""
        # Use in-memory SQLite for testing
        db_manager.database_url = "sqlite:///:memory:"
        db_manager.engine = db_manager.create_engine(db_manager.database_url)
        db_manager.SessionLocal = db_manager.sessionmaker(autocommit=False, autoflush=False, bind=db_manager.engine)
        db_manager.create_tables()
        
        self.collector = FPLDataCollector()
    
    def test_fetch_bootstrap_data(self):
        """Test bootstrap data fetching"""
        mock_response = {
            "elements": [
                {
                    "id": 1,
                    "first_name": "Test",
                    "second_name": "Player",
                    "element_type": 4,
                    "team": 1,
                    "now_cost": 80,
                    "total_points": 50,
                    "selected_by_percent": "5.5",
                    "form": "4.2",
                    "status": "a",
                    "news": ""
                }
            ],
            "teams": [
                {
                    "id": 1,
                    "name": "Test Team",
                    "short_name": "TST",
                    "strength": 4,
                    "strength_overall_home": 4,
                    "strength_overall_away": 3,
                    "strength_attack_home": 4,
                    "strength_attack_away": 3,
                    "strength_defence_home": 4,
                    "strength_defence_away": 3
                }
            ],
            "events": [
                {
                    "id": 1,
                    "name": "Gameweek 1",
                    "deadline_time": "2024-08-16T17:30:00Z",
                    "finished": False,
                    "is_current": True,
                    "is_next": False
                }
            ]
        }
        
        with requests_mock.Mocker() as m:
            m.get(self.collector.base_url + "bootstrap-static/", json=mock_response)
            
            data = self.collector.fetch_bootstrap_data()
            assert data == mock_response
            assert len(data["elements"]) == 1
            assert len(data["teams"]) == 1
            assert len(data["events"]) == 1
    
    def test_save_bootstrap_data_to_db(self):
        """Test saving bootstrap data to database"""
        mock_data = {
            "elements": [
                {
                    "id": 1,
                    "first_name": "Test",
                    "second_name": "Player",
                    "element_type": 4,
                    "team": 1,
                    "now_cost": 80,
                    "total_points": 50,
                    "selected_by_percent": "5.5",
                    "form": "4.2",
                    "status": "a",
                    "news": ""
                }
            ],
            "teams": [
                {
                    "id": 1,
                    "name": "Test Team",
                    "short_name": "TST",
                    "strength": 4,
                    "strength_overall_home": 4,
                    "strength_overall_away": 3,
                    "strength_attack_home": 4,
                    "strength_attack_away": 3,
                    "strength_defence_home": 4,
                    "strength_defence_away": 3
                }
            ],
            "events": [
                {
                    "id": 1,
                    "name": "Gameweek 1",
                    "deadline_time": "2024-08-16T17:30:00Z",
                    "finished": False,
                    "is_current": True,
                    "is_next": False
                }
            ]
        }
        
        self.collector.save_bootstrap_data_to_db(mock_data)
        
        # Verify data was saved
        session = db_manager.get_session()
        try:
            player = session.query(Player).filter_by(id=1).first()
            assert player is not None
            assert player.name == "Test Player"
            assert player.position == 4
            
            team = session.query(Team).filter_by(id=1).first()
            assert team is not None
            assert team.name == "Test Team"
            
            gameweek = session.query(Gameweek).filter_by(id=1).first()
            assert gameweek is not None
            assert gameweek.name == "Gameweek 1"
            assert gameweek.is_current == True
        finally:
            db_manager.close_session(session)
    
    def test_request_retry_logic(self):
        """Test request retry logic on failure"""
        with requests_mock.Mocker() as m:
            # First two requests fail, third succeeds
            m.get(
                self.collector.base_url + "bootstrap-static/",
                [
                    {'status_code': 500},
                    {'status_code': 503},
                    {'json': {'success': True}}
                ]
            )
            
            data = self.collector._make_request(self.collector.base_url + "bootstrap-static/")
            assert data == {'success': True}
    
    def test_request_max_retries_exceeded(self):
        """Test request failure after max retries"""
        with requests_mock.Mocker() as m:
            # All requests fail
            m.get(self.collector.base_url + "bootstrap-static/", status_code=500)
            
            data = self.collector._make_request(self.collector.base_url + "bootstrap-static/", max_retries=2)
            assert data is None

if __name__ == "__main__":
    pytest.main([__file__])



