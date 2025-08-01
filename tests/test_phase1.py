import pytest
from data.collection.fpl_collector import FPLDataCollector
from data.processing.data_processor import DataProcessor
import pandas as pd

def test_fpl_data_collection():
    """Test FPL data collection functionality."""
    collector = FPLDataCollector()
    
    # Test bootstrap data fetch
    bootstrap_data = collector.fetch_bootstrap_data()
    assert 'elements' in bootstrap_data
    assert len(bootstrap_data['elements']) > 500  # Should have 500+ players
    
    # Test fixtures fetch
    fixtures = collector.fetch_fixtures()
    assert not fixtures.empty
    assert 'team_h' in fixtures.columns
    assert 'team_a' in fixtures.columns

def test_data_processing():
    """Test data processing functionality."""
    processor = DataProcessor()
    
    # Test form metrics calculation
    form_metrics = processor.calculate_form_metrics()
    
    if not form_metrics.empty:
        assert 'player_id' in form_metrics.columns
        assert 'avg_points_per_game' in form_metrics.columns
    
    # Test feature creation
    current_gw = 1  # Use gameweek 1 for testing
    features = processor.create_player_features(current_gw)
    
    if not features.empty:
        assert 'player_id' in features.columns
        assert 'position' in features.columns
        assert 'total_points' in features.columns

def test_database_connection():
    """Test database connectivity and schema."""
    from sqlalchemy import create_engine, text
    import os
    
    engine = create_engine(os.getenv('DATABASE_URL'))
    
    # Test basic connectivity
    result = engine.execute(text("SELECT 1")).fetchone()
    assert result[0] == 1
    
    # Test that required tables exist
    tables_query = """
    SELECT table_name 
    FROM information_schema.tables 
    WHERE table_schema = 'public'
    """
    
    tables = pd.read_sql(tables_query, engine)
    required_tables = ['players', 'teams', 'gameweeks', 'fixtures', 'gameweek_stats']
    
    for table in required_tables:
        assert table in tables['table_name'].values

if __name__ == "__main__":
    # Run basic functionality test
    print("Testing Phase 1 implementation...")
    
    try:
        # Test data collection
        collector = FPLDataCollector()
        bootstrap_data = collector.fetch_bootstrap_data()
        print(f"✅ Bootstrap data collected: {len(bootstrap_data['elements'])} players")
        
        # Test data processing
        processor = DataProcessor()
        features = processor.create_player_features(1)
        print(f"✅ Player features created: {len(features)} players")
        
        print("\n🎉 Phase 1 implementation successful!")
        print("\nNext steps:")
        print("1. Review the data collection and processing output")
        print("2. Verify database schema and data quality")
        print("3. Confirm all requirements are met before Phase 2")
        
    except Exception as e:
        print(f"❌ Error in Phase 1 testing: {e}")
        print("Please review the implementation and fix any issues.")
