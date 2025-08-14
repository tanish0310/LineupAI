import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
import logging
from typing import Dict, List, Tuple
import os
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Advanced data processing pipeline that transforms raw FPL data
    into features suitable for machine learning and optimization.
    """
    
    def __init__(self):
        self.engine = create_engine(os.getenv('DATABASE_URL'))
        
    def calculate_form_metrics(self, lookback_gameweeks: int = 10) -> pd.DataFrame:
        """
        Calculate comprehensive form metrics for all players.
        Includes rolling averages, weighted recent performance, and consistency scores.
        """
        try:
            logger.info("Calculating form metrics for all players...")
        
            # Check if gameweek_stats table exists
            try:
                check_query = "SELECT COUNT(*) as count FROM gameweek_stats"
                count_result = pd.read_sql(check_query, self.engine)
            
                if count_result['count'].iloc[0] == 0:
                    logger.warning("No gameweek stats found - returning empty DataFrame for bootstrap phase")
                    return pd.DataFrame(columns=['player_id'])
                
            except Exception as table_error:
                # Table doesn't exist yet - this is normal during bootstrap phase
                logger.warning("gameweek_stats table doesn't exist yet - returning empty DataFrame for bootstrap phase")
                return pd.DataFrame(columns=['player_id'])
        
            # Rest of your existing code for when the table exists and has data...
            query = f"""
            SELECT 
                gs.*,
                p.position,
                p.now_cost,
                f.team_h_difficulty,
                f.team_a_difficulty,
                f.was_home
            FROM gameweek_stats gs
            JOIN players p ON gs.player_id = p.id
            JOIN fixtures f ON gs.fixture_id = f.id
            WHERE gs.gameweek_id > (
                SELECT MAX(id) - {lookback_gameweeks} 
                FROM gameweeks 
                WHERE finished = TRUE
            )
            ORDER BY gs.player_id, gs.gameweek_id
            """
        
            stats_df = pd.read_sql(query, self.engine)
        
            if stats_df.empty:
                logger.warning("No gameweek stats found for form calculation")
                return pd.DataFrame(columns=['player_id'])
        
            # Calculate form metrics for each player
            form_metrics = []
        
            for player_id in stats_df['player_id'].unique():
                player_stats = stats_df[stats_df['player_id'] == player_id].copy()
                player_stats = player_stats.sort_values('gameweek_id')
            
                # Calculate various rolling metrics
                metrics = self._calculate_player_form_metrics(player_stats)
                metrics['player_id'] = player_id
                form_metrics.append(metrics)
        
            form_df = pd.DataFrame(form_metrics)
        
            # Store updated form metrics back to players table
            self._update_player_form_metrics(form_df)
        
            logger.info(f"Successfully calculated form metrics for {len(form_df)} players")
            return form_df
        
        except Exception as e:
            logger.error(f"Error calculating form metrics: {e}")
            # Return empty DataFrame to allow processing to continue
            return pd.DataFrame(columns=['player_id'])

    
    def _calculate_player_form_metrics(self, player_stats: pd.DataFrame) -> Dict:
        """Calculate comprehensive form metrics for a single player."""
        if len(player_stats) == 0:
            return {}
        
        # Basic rolling averages
        metrics = {
            'games_played': len(player_stats),
            'total_points_sum': player_stats['total_points'].sum(),
            'avg_points_per_game': player_stats['total_points'].mean(),
            'avg_minutes_per_game': player_stats['minutes'].mean(),
        }
        
        # Rolling averages for different periods
        for period in [3, 5, 10]:
            recent_games = player_stats.tail(period)
            if len(recent_games) > 0:
                metrics[f'points_{period}gw_avg'] = recent_games['total_points'].mean()
                metrics[f'minutes_{period}gw_avg'] = recent_games['minutes'].mean()
                metrics[f'goals_{period}gw_avg'] = recent_games['goals_scored'].mean()
                metrics[f'assists_{period}gw_avg'] = recent_games['assists'].mean()
        
        # Weighted recent form (more weight to recent games)
        if len(player_stats) >= 3:
            weights = np.array([0.5, 0.3, 0.2])  # Most recent weighted highest
            recent_3 = player_stats.tail(3)['total_points'].values
            if len(recent_3) == 3:
                metrics['weighted_form_3gw'] = np.average(recent_3, weights=weights)
            else:
                metrics['weighted_form_3gw'] = recent_3.mean()
        
        # Consistency metrics
        if len(player_stats) >= 2:
            metrics['points_std'] = player_stats['total_points'].std()
            metrics['consistency_score'] = 1 / (1 + metrics['points_std'])
            
            # Calculate ceiling and floor
            metrics['points_ceiling'] = player_stats['total_points'].max()
            metrics['points_floor'] = player_stats['total_points'].min()
            metrics['upside_potential'] = metrics['points_ceiling'] - metrics['avg_points_per_game']
        
        # Fixture-adjusted performance
        home_games = player_stats[player_stats['was_home'] == True]
        away_games = player_stats[player_stats['was_home'] == False]
        
        if len(home_games) > 0:
            metrics['avg_points_home'] = home_games['total_points'].mean()
        if len(away_games) > 0:
            metrics['avg_points_away'] = away_games['total_points'].mean()
        
        # Difficulty-adjusted performance
        easy_fixtures = player_stats[
            (player_stats['was_home'] & (player_stats['team_h_difficulty'] <= 2)) |
            (~player_stats['was_home'] & (player_stats['team_a_difficulty'] <= 2))
        ]
        hard_fixtures = player_stats[
            (player_stats['was_home'] & (player_stats['team_h_difficulty'] >= 4)) |
            (~player_stats['was_home'] & (player_stats['team_a_difficulty'] >= 4))
        ]
        
        if len(easy_fixtures) > 0:
            metrics['avg_points_easy_fixtures'] = easy_fixtures['total_points'].mean()
        if len(hard_fixtures) > 0:
            metrics['avg_points_hard_fixtures'] = hard_fixtures['total_points'].mean()
        
        # Form trend (improving vs declining)
        if len(player_stats) >= 6:
            first_half = player_stats.head(len(player_stats)//2)['total_points'].mean()
            second_half = player_stats.tail(len(player_stats)//2)['total_points'].mean()
            metrics['form_trend'] = second_half - first_half
        
        # Position-specific metrics
        position = player_stats['position'].iloc[0]
        if position == 1:  # Goalkeeper
            metrics['clean_sheet_rate'] = player_stats['clean_sheets'].mean()
            metrics['saves_per_game'] = player_stats['saves'].mean()
        elif position == 2:  # Defender
            metrics['clean_sheet_rate'] = player_stats['clean_sheets'].mean()
            metrics['attacking_returns_per_game'] = (
                player_stats['goals_scored'] + player_stats['assists']
            ).mean()
        elif position in [3, 4]:  # Midfielder/Forward
            metrics['goals_per_game'] = player_stats['goals_scored'].mean()
            metrics['assists_per_game'] = player_stats['assists'].mean()
            metrics['goal_involvements_per_game'] = (
                player_stats['goals_scored'] + player_stats['assists']
            ).mean()
        
        # Bonus points metrics
        metrics['bonus_per_game'] = player_stats['bonus'].mean()
        metrics['bps_per_game'] = player_stats['bps'].mean()
        
        return metrics
    
    def _update_player_form_metrics(self, form_df: pd.DataFrame):
        """Update players table with calculated form metrics."""
        try:
            with self.engine.connect() as conn:
                for _, row in form_df.iterrows():
                    update_query = """
                    UPDATE players 
                    SET 
                        points_per_game = :ppg,
                        form = :form,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = :player_id
                    """
                
                    conn.execute(text(update_query), {
                        'ppg': row.get('avg_points_per_game', 0),
                        'form': row.get('weighted_form_3gw', row.get('avg_points_per_game', 0)),
                        'player_id': row['player_id']
                    })
            
                conn.commit()  # Commit the transaction
        
        except Exception as e:
            logger.error(f"Error updating player form metrics: {e}")

    
    def generate_fixture_features(self, gameweek_id: int) -> pd.DataFrame:
        """
        Generate comprehensive fixture-based features for the specified gameweek.
        Includes difficulty, rest days, opponent strength, and home/away factors.
        """
        try:
            logger.info(f"Generating fixture features for gameweek {gameweek_id}...")
        
            # Check if fixtures table exists and has data
            try:
                check_query = "SELECT COUNT(*) as count FROM fixtures WHERE gameweek_id = %s"
                count_result = pd.read_sql(check_query, self.engine, params=[gameweek_id])
            
                if count_result['count'].iloc[0] == 0:
                    logger.warning(f"No fixtures found for gameweek {gameweek_id} - returning empty DataFrame for bootstrap phase")
                    return pd.DataFrame(columns=['player_id'])
                
            except Exception as table_error:
                # Table doesn't exist yet - this is normal during bootstrap phase
                logger.warning(f"fixtures table doesn't exist yet - returning empty DataFrame for bootstrap phase")
                return pd.DataFrame(columns=['player_id'])
        
            # Rest of your existing code for when the table exists and has data...
            query = """
            SELECT 
                f.*,
                th.name as home_team_name,
                ta.name as away_team_name,
                th.strength_attack_home,
                th.strength_defence_home,
                ta.strength_attack_away,
                ta.strength_defence_away
            FROM fixtures f
            JOIN teams th ON f.team_h = th.id
            JOIN teams ta ON f.team_a = ta.id
            WHERE f.gameweek_id = %s
            """
        
            fixtures_df = pd.read_sql(query, self.engine, params=[gameweek_id])
        
            if fixtures_df.empty:
                logger.warning(f"No fixtures found for gameweek {gameweek_id}")
                return pd.DataFrame(columns=['player_id'])
        
            # Calculate rest days since last fixture
            fixtures_df = self._calculate_rest_days(fixtures_df, gameweek_id)
        
            # Calculate opponent strength metrics
            fixtures_df = self._calculate_opponent_strength(fixtures_df)
        
            # Generate player-level fixture features
            player_fixtures = self._generate_player_fixture_features(fixtures_df)
        
            logger.info(f"Generated fixture features for {len(player_fixtures)} player-fixture combinations")
            return player_fixtures
        
        except Exception as e:
            logger.error(f"Error generating fixture features: {e}")
            # Return empty DataFrame to allow processing to continue
            return pd.DataFrame(columns=['player_id'])

    
    def _calculate_rest_days(self, fixtures_df: pd.DataFrame, gameweek_id: int) -> pd.DataFrame:
        """Calculate rest days between fixtures for each team."""
        # Get previous gameweek fixtures
        prev_gw_query = """
        SELECT team_h as team_id, kickoff_time, gameweek_id
        FROM fixtures 
        WHERE gameweek_id = :prev_gw AND finished = TRUE
        UNION
        SELECT team_a as team_id, kickoff_time, gameweek_id
        FROM fixtures 
        WHERE gameweek_id = :prev_gw AND finished = TRUE
        """
        
        prev_fixtures = pd.read_sql(
            prev_gw_query, 
            self.engine, 
            params={'prev_gw': gameweek_id - 1}
        )
        
        if not prev_fixtures.empty:
            # Calculate rest days for home teams
            home_rest = self._calculate_team_rest_days(
                fixtures_df, prev_fixtures, 'team_h'
            )
            away_rest = self._calculate_team_rest_days(
                fixtures_df, prev_fixtures, 'team_a'
            )
            
            fixtures_df = fixtures_df.merge(
                home_rest[['team_id', 'rest_days']], 
                left_on='team_h', 
                right_on='team_id', 
                how='left'
            ).drop('team_id', axis=1).rename(columns={'rest_days': 'home_rest_days'})
            
            fixtures_df = fixtures_df.merge(
                away_rest[['team_id', 'rest_days']], 
                left_on='team_a', 
                right_on='team_id', 
                how='left'
            ).drop('team_id', axis=1).rename(columns={'rest_days': 'away_rest_days'})
        
        return fixtures_df
    
    def _calculate_team_rest_days(self, current_fixtures: pd.DataFrame, 
                                prev_fixtures: pd.DataFrame, team_col: str) -> pd.DataFrame:
        """Calculate rest days for teams between gameweeks."""
        rest_days = []
        
        for _, fixture in current_fixtures.iterrows():
            team_id = fixture[team_col]
            current_kickoff = pd.to_datetime(fixture['kickoff_time'])
            
            # Find last fixture for this team
            team_prev = prev_fixtures[prev_fixtures['team_id'] == team_id]
            
            if not team_prev.empty:
                last_kickoff = pd.to_datetime(team_prev['kickoff_time'].max())
                rest_days_calc = (current_kickoff - last_kickoff).days
            else:
                rest_days_calc = 7  # Default assumption
            
            rest_days.append({
                'team_id': team_id,
                'rest_days': rest_days_calc
            })
        
        return pd.DataFrame(rest_days)
    
    def _calculate_opponent_strength(self, fixtures_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate opponent strength metrics for each fixture."""
        # For home teams, opponent is away team
        fixtures_df['home_opponent_attack_strength'] = fixtures_df['strength_attack_away']
        fixtures_df['home_opponent_defence_strength'] = fixtures_df['strength_defence_away']
        
        # For away teams, opponent is home team  
        fixtures_df['away_opponent_attack_strength'] = fixtures_df['strength_attack_home']
        fixtures_df['away_opponent_defence_strength'] = fixtures_df['strength_defence_home']
        
        # Calculate attacking/defensive matchup ratings
        fixtures_df['home_attack_vs_away_defence'] = (
            fixtures_df['strength_attack_home'] - fixtures_df['strength_defence_away']
        )
        fixtures_df['away_attack_vs_home_defence'] = (
            fixtures_df['strength_attack_away'] - fixtures_df['strength_defence_home']
        )
        
        return fixtures_df
    
    def _generate_player_fixture_features(self, fixtures_df: pd.DataFrame) -> pd.DataFrame:
        """Generate player-level features from fixture data."""
        player_fixtures = []
        
        # Get all players for teams playing this gameweek
        teams_playing = list(fixtures_df['team_h'].unique()) + list(fixtures_df['team_a'].unique())
        
        players_query = """
        SELECT id, team, position, web_name, now_cost, status
        FROM players 
        WHERE team = ANY(:teams) AND status = 'a'
        """
        
        players_df = pd.read_sql(
            players_query, 
            self.engine, 
            params={'teams': teams_playing}
        )
        
        for _, player in players_df.iterrows():
            team_id = player['team']
            
            # Find fixture for this team
            home_fixture = fixtures_df[fixtures_df['team_h'] == team_id]
            away_fixture = fixtures_df[fixtures_df['team_a'] == team_id]
            
            if not home_fixture.empty:
                fixture = home_fixture.iloc[0]
                is_home = True
            elif not away_fixture.empty:
                fixture = away_fixture.iloc[0]
                is_home = False
            else:
                continue
            
            # Generate player-specific fixture features
            features = {
                'player_id': player['id'],
                'fixture_id': fixture['id'],
                'gameweek_id': fixture['gameweek_id'],
                'is_home': is_home,
                'kickoff_time': fixture['kickoff_time'],
                'difficulty': fixture['team_h_difficulty'] if is_home else fixture['team_a_difficulty'],
                'opponent_team': fixture['team_a'] if is_home else fixture['team_h'],
                'rest_days': fixture.get('home_rest_days' if is_home else 'away_rest_days', 7),
            }
            
            # Add opponent strength features
            if is_home:
                features.update({
                    'opponent_attack_strength': fixture['strength_attack_away'],
                    'opponent_defence_strength': fixture['strength_defence_away'],
                    'attack_vs_defence_matchup': fixture['home_attack_vs_away_defence'],
                    'home_advantage': 1.1
                })
            else:
                features.update({
                    'opponent_attack_strength': fixture['strength_attack_home'],
                    'opponent_defence_strength': fixture['strength_defence_home'],
                    'attack_vs_defence_matchup': fixture['away_attack_vs_home_defence'],
                    'home_advantage': 0.9
                })
            
            player_fixtures.append(features)
        
        return pd.DataFrame(player_fixtures)
    
    def create_player_features(self, gameweek_id: int) -> pd.DataFrame:
        """Create comprehensive player features - with debugging."""
        try:
            logger.info(f"Creating comprehensive player features for gameweek {gameweek_id}...")
        
            # Test each method individually
            logger.info("Step 1: Getting base features...")
            base_features = self._get_base_player_features()
            logger.info(f"✅ Base features successful: {len(base_features)} players")
        
            logger.info("Step 2: Getting form features...")
            form_features = self.calculate_form_metrics()
            logger.info(f"✅ Form features successful: {len(form_features)} records")
        
            logger.info("Step 3: Getting fixture features...")
            fixture_features = self.generate_fixture_features(gameweek_id)
            logger.info(f"✅ Fixture features successful: {len(fixture_features)} records")
        
            logger.info("Step 4: Getting price features...")
            price_features = self._get_price_ownership_features()
            logger.info(f"✅ Price features successful: {len(price_features)} records")
        
            logger.info("Step 5: Getting availability features...")
            availability_features = self._get_availability_features()
            logger.info(f"✅ Availability features successful: {len(availability_features)} records")
        
            # If we get here, the issue is in the merging
            logger.info("Step 6: Merging features...")
            features = base_features.merge(
                form_features, on='player_id', how='left'
            ).merge(
                fixture_features, on='player_id', how='left'
            ).merge(
                price_features, on='player_id', how='left'
            ).merge(
                availability_features, on='player_id', how='left'
            )
        
            logger.info("Step 7: Finalizing features...")
            features = self._finalize_features(features, gameweek_id)
        
            logger.info(f"✅ Created features for {len(features)} players")
            return features
        
        except Exception as e:
            logger.error(f"❌ Error creating player features: {e}")
            raise

    
    def _get_base_player_features(self) -> pd.DataFrame:
        """Get base player information and stats."""
        query = """
        SELECT 
            id as player_id,
            web_name,
            position,
            team,
            now_cost,
            total_points,
            points_per_game,
            form,
            selected_by_percent,
            status,
            goals_scored,
            assists,
            clean_sheets,
            saves,
            bonus,
            bps,
            influence,
            creativity,
            threat,
            ict_index
        FROM players
        WHERE status = 'a'
        """
    
        return pd.read_sql(query, self.engine)




    
    def _get_price_ownership_features(self) -> pd.DataFrame:
        """Get price change and ownership features."""
        query = """
        SELECT 
            id as player_id,
            now_cost,
            cost_change_event,
            cost_change_start,
            selected_by_percent,
            transfers_in_event,
            transfers_out_event,
            points_per_million,
            value_score,
            total_points
        FROM players
        WHERE status = 'a'
        """
        
        price_df = pd.read_sql(query, self.engine)
        
        # Calculate ownership trends
        price_df['transfer_momentum'] = (
            price_df['transfers_in_event'] - price_df['transfers_out_event']
        )
        
        # Calculate price value metrics
        price_df['price_efficiency'] = price_df['total_points'] / price_df['now_cost']
        
        return price_df[['player_id', 'transfer_momentum', 'price_efficiency']]
    
    def _get_availability_features(self) -> pd.DataFrame:
        """Get injury and availability features."""
        try:
            # Try the full query with team_news first
            query = """
            SELECT 
                p.id as player_id,
                p.chance_of_playing_this_round,
                p.chance_of_playing_next_round,
                p.news,
                COALESCE(tn.injury_status, 'available') as injury_status,
                COALESCE(tn.confidence_score, 1.0) as availability_confidence
            FROM players p
            LEFT JOIN (
                SELECT DISTINCT ON (player_id) 
                    player_id, injury_status, confidence_score
                FROM team_news 
                ORDER BY player_id, created_at DESC
            ) tn ON p.id = tn.player_id
            WHERE p.status = 'a'
            """
        
            availability_df = pd.read_sql(query, self.engine)
        
        except Exception as table_error:
            # team_news table doesn't exist yet - use simplified query
            logger.warning("team_news table doesn't exist yet - using simplified availability features for bootstrap phase")
        
            simple_query = """
            SELECT 
                p.id as player_id,
                p.chance_of_playing_this_round,
                p.chance_of_playing_next_round,
                p.news,
                'available' as injury_status,
                1.0 as availability_confidence
            FROM players p
            WHERE p.status = 'a'
            """
        
            availability_df = pd.read_sql(simple_query, self.engine)
    
        # Calculate availability score
        availability_df['availability_score'] = availability_df.apply(
            self._calculate_availability_score, axis=1
        )
    
        return availability_df[['player_id', 'availability_score']]

    
    def _calculate_availability_score(self, row) -> float:
        """Calculate comprehensive availability score for a player."""
        base_score = 1.0
        
        # Reduce based on chance of playing
        if pd.notna(row['chance_of_playing_this_round']):
            base_score *= (row['chance_of_playing_this_round'] / 100)
        
        # Reduce based on injury status
        injury_status = row.get('injury_status', 'available')
        if injury_status == 'doubtful':
            base_score *= 0.7
        elif injury_status == 'injured':
            base_score *= 0.3
        elif injury_status == 'suspended':
            base_score *= 0.0
        
        # Factor in news sentiment
        news = str(row.get('news', '')).lower()
        if any(word in news for word in ['injured', 'doubt', 'suspended']):
            base_score *= 0.8
        
        return max(0.0, min(1.0, base_score))
    
    def _finalize_features(self, features: pd.DataFrame, gameweek_id: int) -> pd.DataFrame:
        """Finalize feature set with missing value handling and feature engineering."""
    
        # Convert numeric columns to proper data types
        numeric_columns = [
            'total_points', 'now_cost', 'form', 'points_per_game', 
            'selected_by_percent', 'goals_scored', 'assists', 'clean_sheets', 
            'saves', 'bonus', 'bps', 'influence', 'creativity', 'threat', 'ict_index'
        ]
    
        for col in numeric_columns:
            if col in features.columns:
                features[col] = pd.to_numeric(features[col], errors='coerce')
    
        # Fill missing values after conversion
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        features[numeric_cols] = features[numeric_cols].fillna(0)
    
        # Add gameweek identifier
        features['gameweek_id'] = gameweek_id
    
        # Create interaction features - now safe to divide
        features['points_per_cost'] = features['total_points'] / features['now_cost']
        features['form_vs_price'] = features.get('form', 0) / features['now_cost']
    
        # Position-specific feature adjustments
        features = self._create_position_specific_features(features)
    
        return features

    
    def _create_position_specific_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Create position-specific features and adjustments."""
        # Goalkeeper features
        gk_mask = features['position'] == 1
        features.loc[gk_mask, 'primary_stat'] = features.loc[gk_mask, 'saves']
        features.loc[gk_mask, 'secondary_stat'] = features.loc[gk_mask, 'clean_sheets']
        
        # Defender features
        def_mask = features['position'] == 2
        features.loc[def_mask, 'primary_stat'] = features.loc[def_mask, 'clean_sheets']
        features.loc[def_mask, 'secondary_stat'] = (
            features.loc[def_mask, 'goals_scored'] + features.loc[def_mask, 'assists']
        )
        
        # Midfielder features
        mid_mask = features['position'] == 3
        features.loc[mid_mask, 'primary_stat'] = (
            features.loc[mid_mask, 'goals_scored'] + features.loc[mid_mask, 'assists']
        )
        features.loc[mid_mask, 'secondary_stat'] = features.loc[mid_mask, 'creativity']
        
        # Forward features
        fwd_mask = features['position'] == 4
        features.loc[fwd_mask, 'primary_stat'] = features.loc[fwd_mask, 'goals_scored']
        features.loc[fwd_mask, 'secondary_stat'] = features.loc[fwd_mask, 'threat']
        
        return features

