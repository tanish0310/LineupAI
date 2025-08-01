import requests
import pandas as pd
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
from sqlalchemy import create_engine
from bs4 import BeautifulSoup
import os
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FPLDataCollector:
    """
    Comprehensive FPL data collection system that fetches all necessary
    data from the FPL API and external sources.
    """
    
    def __init__(self):
        self.base_url = os.getenv('FPL_API_BASE', 'https://fantasy.premierleague.com/api/')
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.engine = create_engine(os.getenv('DATABASE_URL'))
        
    def fetch_bootstrap_data(self) -> Dict:
        """
        Fetch core FPL data including all players, teams, and gameweeks.
        This is the foundation dataset for the entire system.
        """
        try:
            logger.info("Fetching bootstrap data from FPL API...")
            response = self.session.get(f"{self.base_url}bootstrap-static/")
            response.raise_for_status()
            
            data = response.json()
            
            # Extract and process different data types
            players_data = self._process_players_data(data['elements'])
            teams_data = self._process_teams_data(data['teams'])
            gameweeks_data = self._process_gameweeks_data(data['events'])
            positions_data = data['element_types']
            
            # Store in database
            self._store_bootstrap_data(players_data, teams_data, gameweeks_data, positions_data)
            
            logger.info(f"Successfully fetched data for {len(players_data)} players")
            return data
            
        except requests.RequestException as e:
            logger.error(f"Error fetching bootstrap data: {e}")
            raise
    
    def _process_players_data(self, players_raw: List[Dict]) -> pd.DataFrame:
        """Process raw player data into structured format with calculated metrics."""
        players_df = pd.DataFrame(players_raw)
        
        # Calculate additional metrics
        players_df['points_per_million'] = players_df['total_points'] / (players_df['now_cost'] / 10)
        players_df['form_rating'] = pd.to_numeric(players_df['form'], errors='coerce')
        players_df['availability_score'] = players_df.apply(self._calculate_availability, axis=1)
        players_df['value_score'] = self._calculate_value_score(players_df)
        
        # Clean and standardize data
        players_df['price'] = players_df['now_cost'] / 10  # Convert to actual price
        players_df['position'] = players_df['element_type']
        players_df['ownership'] = pd.to_numeric(players_df['selected_by_percent'], errors='coerce')
        
        return players_df
    
    def _calculate_availability(self, player_row) -> float:
        """Calculate player availability score based on injury status and news."""
        base_score = 1.0
        
        # Reduce score based on injury status
        if player_row.get('chance_of_playing_next_round'):
            chance = player_row['chance_of_playing_next_round']
            if chance is not None:
                base_score = chance / 100
        
        # Further reduction for specific injury news
        news = str(player_row.get('news', '')).lower()
        if any(word in news for word in ['injured', 'doubt', 'suspended']):
            base_score *= 0.5
        
        return base_score
    
    def _calculate_value_score(self, players_df: pd.DataFrame) -> pd.Series:
        """Calculate relative value score for each player within their position."""
        value_scores = pd.Series(index=players_df.index, dtype=float)
        
        for position in players_df['element_type'].unique():
            position_mask = players_df['element_type'] == position
            position_players = players_df[position_mask]
            
            # Calculate value as points per million relative to position average
            position_avg_ppm = position_players['points_per_million'].mean()
            relative_value = position_players['points_per_million'] / position_avg_ppm
            
            value_scores[position_mask] = relative_value
        
        return value_scores
    
    def fetch_fixtures(self) -> pd.DataFrame:
        """
        Fetch fixture data including difficulty ratings and scheduling.
        Critical for predicting player performance based on upcoming opponents.
        """
        try:
            logger.info("Fetching fixtures data...")
            response = self.session.get(f"{self.base_url}fixtures/")
            response.raise_for_status()
            
            fixtures = pd.DataFrame(response.json())
            
            # Process fixture data
            fixtures['kickoff_time'] = pd.to_datetime(fixtures['kickoff_time'])
            fixtures['home_advantage'] = 1.1  # Statistical home advantage factor
            
            # Calculate team strength metrics
            fixtures = self._add_team_strength_metrics(fixtures)
            
            # Store in database
            fixtures.to_sql('fixtures', self.engine, if_exists='replace', index=False)
            
            logger.info(f"Successfully fetched {len(fixtures)} fixtures")
            return fixtures
            
        except requests.RequestException as e:
            logger.error(f"Error fetching fixtures: {e}")
            raise
    
    def _add_team_strength_metrics(self, fixtures_df: pd.DataFrame) -> pd.DataFrame:
        """Add team strength metrics to fixtures for better predictions."""
        # This would typically use historical data to calculate team strengths
        # For now, using FPL's difficulty ratings as base
        fixtures_df['home_team_strength'] = 6 - fixtures_df['team_h_difficulty']
        fixtures_df['away_team_strength'] = 6 - fixtures_df['team_a_difficulty']
        
        return fixtures_df
    
    def fetch_gameweek_live_data(self, gameweek: int) -> Dict:
        """
        Fetch live gameweek data including player points, bonus, and performance.
        Essential for updating predictions and analyzing actual vs predicted performance.
        """
        try:
            logger.info(f"Fetching live data for gameweek {gameweek}...")
            response = self.session.get(f"{self.base_url}event/{gameweek}/live/")
            response.raise_for_status()
            
            live_data = response.json()
            
            # Process player performance data
            player_stats = []
            for player_data in live_data['elements']:
                stats = {
                    'player_id': player_data['id'],
                    'gameweek_id': gameweek,
                    'total_points': player_data['stats']['total_points'],
                    'minutes': player_data['stats']['minutes'],
                    'goals_scored': player_data['stats']['goals_scored'],
                    'assists': player_data['stats']['assists'],
                    'clean_sheets': player_data['stats']['clean_sheets'],
                    'saves': player_data['stats']['saves'],
                    'bonus': player_data['stats']['bonus'],
                    'bps': player_data['stats']['bps'],
                    'influence': float(player_data['stats']['influence']),
                    'creativity': float(player_data['stats']['creativity']),
                    'threat': float(player_data['stats']['threat']),
                    'ict_index': float(player_data['stats']['ict_index']),
                    'timestamp': datetime.now()
                }
                player_stats.append(stats)
            
            # Store in database
            stats_df = pd.DataFrame(player_stats)
            stats_df.to_sql('gameweek_stats', self.engine, if_exists='append', index=False)
            
            logger.info(f"Successfully fetched live data for {len(player_stats)} players")
            return live_data
            
        except requests.RequestException as e:
            logger.error(f"Error fetching gameweek live data: {e}")
            raise
    
    def fetch_player_history(self, player_id: int) -> pd.DataFrame:
        """
        Fetch detailed historical data for a specific player.
        Used for advanced feature engineering and form analysis.
        """
        try:
            response = self.session.get(f"{self.base_url}element-summary/{player_id}/")
            response.raise_for_status()
            
            data = response.json()
            
            # Process historical gameweek data
            history_df = pd.DataFrame(data['history'])
            
            if not history_df.empty:
                # Calculate rolling averages and trends
                history_df = self._calculate_player_trends(history_df)
            
            return history_df
            
        except requests.RequestException as e:
            logger.error(f"Error fetching player {player_id} history: {e}")
            return pd.DataFrame()
    
    def _calculate_player_trends(self, history_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate rolling trends and form metrics for a player."""
        # Sort by round (gameweek)
        history_df = history_df.sort_values('round')
        
        # Calculate rolling averages
        history_df['points_3gw_avg'] = history_df['total_points'].rolling(3, min_periods=1).mean()
        history_df['points_5gw_avg'] = history_df['total_points'].rolling(5, min_periods=1).mean()
        history_df['minutes_3gw_avg'] = history_df['minutes'].rolling(3, min_periods=1).mean()
        
        # Calculate form trends (weighted recent games more heavily)
        weights_3gw = [0.5, 0.3, 0.2]  # Most recent game weighted highest
        history_df['weighted_form_3gw'] = history_df['total_points'].rolling(3, min_periods=1).apply(
            lambda x: sum(x.iloc[i] * weights_3gw[i] for i in range(len(x))) / sum(weights_3gw[:len(x)])
        )
        
        # Calculate consistency metrics
        history_df['points_std_5gw'] = history_df['total_points'].rolling(5, min_periods=2).std()
        history_df['consistency_score'] = 1 / (1 + history_df['points_std_5gw'].fillna(0))
        
        return history_df
    
    def fetch_team_news(self) -> Dict:
        """
        Scrape injury reports and team news from multiple sources.
        Critical for updating player availability and predicting lineups.
        """
        team_news = {}
        
        try:
            # This would scrape from official team websites, sports news sites
            # For demo purposes, using a placeholder structure
            sources = [
                'https://www.premierleague.com/news',
                'https://www.skysports.com/football/news'
            ]
            
            for source in sources:
                try:
                    response = self.session.get(source, timeout=10)
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Extract injury news (this would be source-specific)
                    news_items = self._extract_injury_news(soup, source)
                    team_news.update(news_items)
                    
                except Exception as e:
                    logger.warning(f"Could not fetch team news from {source}: {e}")
                    continue
            
            # Store team news in database
            self._store_team_news(team_news)
            
            logger.info(f"Successfully fetched team news from {len(sources)} sources")
            return team_news
            
        except Exception as e:
            logger.error(f"Error fetching team news: {e}")
            return {}
    
    def _extract_injury_news(self, soup: BeautifulSoup, source: str) -> Dict:
        """Extract injury news from scraped website content."""
        # This would contain source-specific parsing logic
        # For now, returning empty dict as placeholder
        return {}
    
    def _store_bootstrap_data(self, players_df: pd.DataFrame, teams_df: pd.DataFrame, 
                            gameweeks_df: pd.DataFrame, positions_data: List[Dict]):
        """Store all bootstrap data in the database."""
        try:
            # Store players
            players_df.to_sql('players', self.engine, if_exists='replace', index=False)
            
            # Store teams
            teams_df = pd.DataFrame(teams_data) if isinstance(teams_data, list) else teams_df
            teams_df.to_sql('teams', self.engine, if_exists='replace', index=False)
            
            # Store gameweeks
            gameweeks_df = pd.DataFrame(gameweeks_data) if isinstance(gameweeks_data, list) else gameweeks_df
            gameweeks_df.to_sql('gameweeks', self.engine, if_exists='replace', index=False)
            
            # Store positions
            positions_df = pd.DataFrame(positions_data)
            positions_df.to_sql('positions', self.engine, if_exists='replace', index=False)
            
            logger.info("Successfully stored all bootstrap data in database")
            
        except Exception as e:
            logger.error(f"Error storing bootstrap data: {e}")
            raise
    
    def _store_team_news(self, team_news: Dict):
        """Store team news and injury updates in database."""
        if not team_news:
            return
            
        try:
            news_records = []
            for player_id, news_data in team_news.items():
                record = {
                    'player_id': player_id,
                    'news_text': news_data.get('text', ''),
                    'injury_status': news_data.get('injury_status', 'available'),
                    'expected_return': news_data.get('expected_return'),
                    'source': news_data.get('source', ''),
                    'timestamp': datetime.now()
                }
                news_records.append(record)
            
            news_df = pd.DataFrame(news_records)
            news_df.to_sql('team_news', self.engine, if_exists='append', index=False)
            
        except Exception as e:
            logger.error(f"Error storing team news: {e}")
    
    def _process_teams_data(self, teams_raw: List[Dict]) -> List[Dict]:
        """Process and enhance teams data."""
        return teams_raw
    
    def _process_gameweeks_data(self, gameweeks_raw: List[Dict]) -> List[Dict]:
        """Process and enhance gameweeks data."""
        return gameweeks_raw
    
    def get_current_gameweek(self) -> int:
        """Get the current active gameweek."""
        try:
            bootstrap_data = self.fetch_bootstrap_data()
            
            for gw in bootstrap_data['events']:
                if gw['is_current']:
                    return gw['id']
            
            # If no current gameweek, return next upcoming
            for gw in bootstrap_data['events']:
                if not gw['finished']:
                    return gw['id']
                    
            return 1  # Fallback
            
        except Exception as e:
            logger.error(f"Error getting current gameweek: {e}")
            return 1
    
    def fetch_all_data(self):
        """Fetch all necessary data in the correct order."""
        try:
            logger.info("Starting comprehensive data collection...")
            
            # 1. Fetch core data
            bootstrap_data = self.fetch_bootstrap_data()
            
            # 2. Fetch fixtures
            fixtures = self.fetch_fixtures()
            
            # 3. Fetch current gameweek live data
            current_gw = self.get_current_gameweek()
            if current_gw > 1:
                live_data = self.fetch_gameweek_live_data(current_gw)
            
            # 4. Fetch team news
            team_news = self.fetch_team_news()
            
            logger.info("Comprehensive data collection completed successfully")
            
        except Exception as e:
            logger.error(f"Error in comprehensive data collection: {e}")
            raise




