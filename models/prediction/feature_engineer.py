import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sqlalchemy import create_engine, text
import logging
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Advanced feature engineering system that creates position-specific features
    for optimal ML model performance across all FPL positions.
    """
    
    def __init__(self):
        self.engine = create_engine(os.getenv('DATABASE_URL'))
        
        # Position-specific feature configurations
        self.position_configs = {
            1: 'goalkeeper',  # GK
            2: 'defender',    # DEF  
            3: 'midfielder',  # MID
            4: 'forward'      # FWD
        }
        
        # Position-specific primary stats for feature importance
        self.position_primary_stats = {
            'goalkeeper': ['saves', 'clean_sheets', 'saves_per_game', 'goals_conceded'],
            'defender': ['clean_sheets', 'goals_scored', 'assists', 'bonus'],
            'midfielder': ['goals_scored', 'assists', 'creativity', 'ict_index'],
            'forward': ['goals_scored', 'threat', 'penalties_missed', 'ict_index']
        }
    
    def goalkeeper_features(self, player_data: pd.DataFrame, gameweek_id: int) -> Dict:
        """
        Generate goalkeeper-specific features focusing on clean sheets, saves, and defensive metrics.
        """
        try:
            player_id = player_data['player_id'].iloc[0]
            
            # Basic goalkeeper metrics
            base_features = {
                'player_id': player_id,
                'position': 1,
                'gameweek_id': gameweek_id,
                'now_cost': player_data['now_cost'].iloc[0],
                'total_points': player_data['total_points'].iloc[0],
                'form': player_data['form'].iloc[0],
                'minutes_played_pct': self._calculate_minutes_percentage(player_id),
            }
            
            # Clean sheet probability based on team defensive strength and fixtures
            base_features['clean_sheet_probability'] = self._calculate_clean_sheet_probability(
                player_id, gameweek_id
            )
            
            # Saves prediction based on opponent attacking strength
            base_features['expected_saves'] = self._calculate_expected_saves(
                player_id, gameweek_id
            )
            
            # Historical performance metrics
            historical_stats = self._get_goalkeeper_historical_stats(player_id)
            base_features.update(historical_stats)
            
            # Team defensive metrics
            team_defense = self._get_team_defensive_metrics(player_data['team'].iloc[0])
            base_features.update(team_defense)
            
            # Fixture-specific adjustments
            fixture_features = self._get_goalkeeper_fixture_features(player_id, gameweek_id)
            base_features.update(fixture_features)
            
            # Advanced goalkeeper metrics
            advanced_features = self._calculate_advanced_gk_features(player_id)
            base_features.update(advanced_features)
            
            return base_features
            
        except Exception as e:
            logger.error(f"Error generating goalkeeper features for player {player_data['player_id'].iloc[0]}: {e}")
            return {}
    
    def defender_features(self, player_data: pd.DataFrame, gameweek_id: int) -> Dict:
        """
        Generate defender-specific features focusing on clean sheets, attacking threat, and bonus potential.
        """
        try:
            player_id = player_data['player_id'].iloc[0]
            
            base_features = {
                'player_id': player_id,
                'position': 2,
                'gameweek_id': gameweek_id,
                'now_cost': player_data['now_cost'].iloc[0],
                'total_points': player_data['total_points'].iloc[0],
                'form': player_data['form'].iloc[0],
            }
            
            # Clean sheet probability (primary for defenders)
            base_features['clean_sheet_probability'] = self._calculate_clean_sheet_probability(
                player_id, gameweek_id
            )
            
            # Attacking threat from defenders
            base_features['attacking_threat'] = self._calculate_defender_attacking_threat(player_id)
            
            # Bonus point frequency (defenders often get bonus for clean sheets)
            base_features['bonus_point_frequency'] = self._calculate_bonus_frequency(player_id)
            
            # Set piece involvement
            base_features['set_piece_threat'] = self._calculate_set_piece_involvement(player_id)
            
            # Historical defensive stats
            defensive_stats = self._get_defender_historical_stats(player_id)
            base_features.update(defensive_stats)
            
            # Team context features
            team_features = self._get_team_context_features(player_data['team'].iloc[0], gameweek_id)
            base_features.update(team_features)
            
            # Position-specific advanced metrics
            advanced_features = self._calculate_advanced_defender_features(player_id)
            base_features.update(advanced_features)
            
            return base_features
            
        except Exception as e:
            logger.error(f"Error generating defender features for player {player_data['player_id'].iloc[0]}: {e}")
            return {}
    
    def midfielder_features(self, player_data: pd.DataFrame, gameweek_id: int) -> Dict:
        """
        Generate midfielder-specific features focusing on goals, assists, creativity, and overall involvement.
        """
        try:
            player_id = player_data['player_id'].iloc[0]
            
            base_features = {
                'player_id': player_id,
                'position': 3,
                'gameweek_id': gameweek_id,
                'now_cost': player_data['now_cost'].iloc[0],
                'total_points': player_data['total_points'].iloc[0],
                'form': player_data['form'].iloc[0],
            }
            
            # Goal scoring potential
            base_features['goals_per_game_avg'] = self._calculate_goals_per_game(player_id)
            base_features['expected_goals_per_game'] = self._calculate_expected_goals(player_id)
            
            # Assist potential
            base_features['assists_per_game_avg'] = self._calculate_assists_per_game(player_id)
            base_features['expected_assists_per_game'] = self._calculate_expected_assists(player_id)
            
            # Creativity index (crucial for midfielders)
            base_features['creativity_index'] = self._calculate_creativity_index(player_id)
            
            # Overall ICT involvement
            base_features['ict_involvement'] = self._calculate_ict_involvement(player_id)
            
            # Penalty taking probability
            base_features['penalty_taker'] = self._is_penalty_taker(player_id)
            
            # Free kick involvement
            base_features['free_kick_taker'] = self._is_free_kick_taker(player_id)
            
            # Historical midfielder stats
            midfielder_stats = self._get_midfielder_historical_stats(player_id)
            base_features.update(midfielder_stats)
            
            # Team attacking context
            attacking_context = self._get_team_attacking_context(player_data['team'].iloc[0], gameweek_id)
            base_features.update(attacking_context)
            
            # Advanced midfielder metrics
            advanced_features = self._calculate_advanced_midfielder_features(player_id)
            base_features.update(advanced_features)
            
            return base_features
            
        except Exception as e:
            logger.error(f"Error generating midfielder features for player {player_data['player_id'].iloc[0]}: {e}")
            return {}
    
    def forward_features(self, player_data: pd.DataFrame, gameweek_id: int) -> Dict:
        """
        Generate forward-specific features focusing on goals, shots, and attacking output.
        """
        try:
            player_id = player_data['player_id'].iloc[0]
            
            base_features = {
                'player_id': player_id,
                'position': 4,
                'gameweek_id': gameweek_id,
                'now_cost': player_data['now_cost'].iloc[0],
                'total_points': player_data['total_points'].iloc[0],
                'form': player_data['form'].iloc[0],
            }
            
            # Primary forward metric - goal scoring
            base_features['goals_per_game_avg'] = self._calculate_goals_per_game(player_id)
            base_features['expected_goals_per_game'] = self._calculate_expected_goals(player_id)
            
            # Secondary attacking metrics
            base_features['assists_per_game_avg'] = self._calculate_assists_per_game(player_id)
            base_features['shots_per_game'] = self._calculate_shots_per_game(player_id)
            
            # Penalty involvement
            base_features['penalty_taker'] = self._is_penalty_taker(player_id)
            base_features['penalty_conversion_rate'] = self._calculate_penalty_conversion(player_id)
            
            # Threat index (crucial for forwards)
            base_features['threat_index'] = self._calculate_threat_index(player_id)
            
            # Minutes and rotation risk
            base_features['minutes_certainty'] = self._calculate_minutes_certainty(player_id)
            base_features['rotation_risk'] = self._calculate_rotation_risk(player_id)
            
            # Historical forward stats
            forward_stats = self._get_forward_historical_stats(player_id)
            base_features.update(forward_stats)
            
            # Opposition defensive analysis
            opposition_analysis = self._get_opposition_defensive_analysis(player_id, gameweek_id)
            base_features.update(opposition_analysis)
            
            # Advanced forward metrics
            advanced_features = self._calculate_advanced_forward_features(player_id)
            base_features.update(advanced_features)
            
            return base_features
            
        except Exception as e:
            logger.error(f"Error generating forward features for player {player_data['player_id'].iloc[0]}: {e}")
            return {}
    
    # Helper methods for feature calculations
    
    def _calculate_minutes_percentage(self, player_id: int) -> float:
        """Calculate percentage of available minutes played by player."""
        try:
            query = """
            SELECT AVG(minutes) as avg_minutes
            FROM gameweek_stats 
            WHERE player_id = :player_id
            AND gameweek_id > (SELECT MAX(id) - 10 FROM gameweeks WHERE finished = TRUE)
            """
            
            result = pd.read_sql(query, self.engine, params={'player_id': player_id})
            avg_minutes = result['avg_minutes'].iloc[0] if not result.empty else 0
            
            return min(avg_minutes / 90.0, 1.0)  # Cap at 100%
            
        except Exception:
            return 0.5  # Default assumption
    
    def _calculate_clean_sheet_probability(self, player_id: int, gameweek_id: int) -> float:
        """Calculate probability of clean sheet based on team defense and opponent."""
        try:
            # Get team defensive strength and opponent attacking strength
            query = """
            SELECT 
                th.strength_defence_home,
                ta.strength_attack_away,
                f.team_h_difficulty,
                f.team_a_difficulty,
                p.team = f.team_h as is_home
            FROM players p
            JOIN fixtures f ON (p.team = f.team_h OR p.team = f.team_a)
            JOIN teams th ON f.team_h = th.id
            JOIN teams ta ON f.team_a = ta.id
            WHERE p.id = :player_id 
            AND f.gameweek_id = :gameweek_id
            """
            
            result = pd.read_sql(query, self.engine, params={
                'player_id': player_id, 
                'gameweek_id': gameweek_id
            })
            
            if result.empty:
                return 0.3  # Default probability
            
            is_home = result['is_home'].iloc[0]
            
            if is_home:
                defense_strength = result['strength_defence_home'].iloc[0]
                attack_strength = result['strength_attack_away'].iloc[0]
                difficulty = result['team_h_difficulty'].iloc[0]
            else:
                # Get away team defensive strength
                query_away = """
                SELECT strength_defence_away 
                FROM teams 
                WHERE id = (SELECT team FROM players WHERE id = :player_id)
                """
                away_defense = pd.read_sql(query_away, self.engine, params={'player_id': player_id})
                defense_strength = away_defense['strength_defence_away'].iloc[0]
                attack_strength = result['strength_attack_home'].iloc[0]
                difficulty = result['team_a_difficulty'].iloc[0]
            
            # Calculate probability based on strength differential and home advantage
            strength_diff = defense_strength - attack_strength
            home_bonus = 0.1 if is_home else 0
            
            # Base probability adjusted by difficulty
            base_prob = 0.5 - (difficulty - 3) * 0.1
            strength_adjustment = strength_diff * 0.05
            
            clean_sheet_prob = base_prob + strength_adjustment + home_bonus
            
            return max(0.0, min(1.0, clean_sheet_prob))
            
        except Exception:
            return 0.3  # Default fallback
    
    def _calculate_expected_saves(self, player_id: int, gameweek_id: int) -> float:
        """Calculate expected saves based on opponent attacking strength."""
        try:
            # Get opponent attacking metrics
            query = """
            SELECT 
                CASE 
                    WHEN p.team = f.team_h THEN ta.strength_attack_away
                    ELSE th.strength_attack_home
                END as opponent_attack_strength,
                p.team = f.team_h as is_home
            FROM players p
            JOIN fixtures f ON (p.team = f.team_h OR p.team = f.team_a)
            JOIN teams th ON f.team_h = th.id
            JOIN teams ta ON f.team_a = ta.id
            WHERE p.id = :player_id 
            AND f.gameweek_id = :gameweek_id
            """
            
            result = pd.read_sql(query, self.engine, params={
                'player_id': player_id,
                'gameweek_id': gameweek_id
            })
            
            if result.empty:
                return 3.0  # Default expected saves
            
            opponent_attack = result['opponent_attack_strength'].iloc[0]
            is_home = result['is_home'].iloc[0]
            
            # Base saves adjusted by opponent strength
            base_saves = 3.0
            attack_adjustment = (opponent_attack - 3) * 0.5
            home_adjustment = -0.3 if is_home else 0.3  # Home teams face fewer shots
            
            expected_saves = base_saves + attack_adjustment + home_adjustment
            
            return max(0.0, expected_saves)
            
        except Exception:
            return 3.0
    
    def _calculate_defender_attacking_threat(self, player_id: int) -> float:
        """Calculate attacking threat for defenders based on historical data."""
        try:
            query = """
            SELECT 
                AVG(goals_scored + assists) as goal_involvements_per_game,
                AVG(bps) as avg_bps
            FROM gameweek_stats gs
            WHERE gs.player_id = :player_id
            AND gs.gameweek_id > (SELECT MAX(id) - 10 FROM gameweeks WHERE finished = TRUE)
            AND gs.minutes > 60
            """
            
            result = pd.read_sql(query, self.engine, params={'player_id': player_id})
            
            if result.empty:
                return 0.1  # Low default for defenders
            
            goal_involvements = result['goal_involvements_per_game'].iloc[0] or 0
            avg_bps = result['avg_bps'].iloc[0] or 0
            
            # Combine goal involvements with bonus point system score
            attacking_threat = goal_involvements + (avg_bps / 100)
            
            return attacking_threat
            
        except Exception:
            return 0.1
    
    def _calculate_bonus_frequency(self, player_id: int) -> float:
        """Calculate how often player receives bonus points."""
        try:
            query = """
            SELECT 
                AVG(CASE WHEN bonus > 0 THEN 1 ELSE 0 END) as bonus_frequency,
                AVG(bonus) as avg_bonus
            FROM gameweek_stats
            WHERE player_id = :player_id
            AND gameweek_id > (SELECT MAX(id) - 10 FROM gameweeks WHERE finished = TRUE)
            AND minutes > 0
            """
            
            result = pd.read_sql(query, self.engine, params={'player_id': player_id})
            
            if result.empty:
                return 0.2
            
            return result['bonus_frequency'].iloc[0] or 0.2
            
        except Exception:
            return 0.2
    
    def _calculate_set_piece_involvement(self, player_id: int) -> float:
        """Estimate set piece involvement for defenders."""
        try:
            # This would ideally use set piece data, but we'll estimate from goals/assists
            query = """
            SELECT 
                AVG(goals_scored) as avg_goals,
                COUNT(*) as games_played
            FROM gameweek_stats
            WHERE player_id = :player_id
            AND gameweek_id > (SELECT MAX(id) - 20 FROM gameweeks WHERE finished = TRUE)
            AND minutes > 60
            """
            
            result = pd.read_sql(query, self.engine, params={'player_id': player_id})
            
            if result.empty:
                return 0.1
            
            avg_goals = result['avg_goals'].iloc[0] or 0
            
            # Defenders scoring regularly likely have set piece involvement
            set_piece_score = min(avg_goals * 5, 1.0)  # Scale to 0-1
            
            return set_piece_score
            
        except Exception:
            return 0.1
    
    def _calculate_goals_per_game(self, player_id: int) -> float:
        """Calculate average goals per game for midfielder/forward."""
        try:
            query = """
            SELECT AVG(goals_scored) as avg_goals
            FROM gameweek_stats
            WHERE player_id = :player_id
            AND gameweek_id > (SELECT MAX(id) - 10 FROM gameweeks WHERE finished = TRUE)
            AND minutes > 30
            """
            
            result = pd.read_sql(query, self.engine, params={'player_id': player_id})
            
            return result['avg_goals'].iloc[0] if not result.empty else 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_expected_goals(self, player_id: int) -> float:
        """Calculate expected goals (would need xG data, using approximation)."""
        try:
            # Approximate xG from shots and goals
            query = """
            SELECT 
                AVG(goals_scored) as avg_goals,
                AVG(threat) as avg_threat
            FROM gameweek_stats
            WHERE player_id = :player_id
            AND gameweek_id > (SELECT MAX(id) - 10 FROM gameweeks WHERE finished = TRUE)
            AND minutes > 30
            """
            
            result = pd.read_sql(query, self.engine, params={'player_id': player_id})
            
            if result.empty:
                return 0.0
            
            # Use threat as proxy for expected goals
            avg_threat = result['avg_threat'].iloc[0] or 0
            expected_goals = avg_threat / 100  # Rough conversion
            
            return expected_goals
            
        except Exception:
            return 0.0
    
    def _calculate_assists_per_game(self, player_id: int) -> float:
        """Calculate average assists per game."""
        try:
            query = """
            SELECT AVG(assists) as avg_assists
            FROM gameweek_stats
            WHERE player_id = :player_id
            AND gameweek_id > (SELECT MAX(id) - 10 FROM gameweeks WHERE finished = TRUE)
            AND minutes > 30
            """
            
            result = pd.read_sql(query, self.engine, params={'player_id': player_id})
            
            return result['avg_assists'].iloc[0] if not result.empty else 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_expected_assists(self, player_id: int) -> float:
        """Calculate expected assists using creativity as proxy."""
        try:
            query = """
            SELECT AVG(creativity) as avg_creativity
            FROM gameweek_stats
            WHERE player_id = :player_id
            AND gameweek_id > (SELECT MAX(id) - 10 FROM gameweeks WHERE finished = TRUE)
            AND minutes > 30
            """
            
            result = pd.read_sql(query, self.engine, params={'player_id': player_id})
            
            if result.empty:
                return 0.0
            
            # Use creativity as proxy for expected assists
            avg_creativity = result['avg_creativity'].iloc[0] or 0
            expected_assists = avg_creativity / 150  # Rough conversion
            
            return expected_assists
            
        except Exception:
            return 0.0
    
    def _calculate_creativity_index(self, player_id: int) -> float:
        """Calculate creativity index for midfielders."""
        try:
            query = """
            SELECT AVG(creativity) as avg_creativity
            FROM gameweek_stats
            WHERE player_id = :player_id
            AND gameweek_id > (SELECT MAX(id) - 10 FROM gameweeks WHERE finished = TRUE)
            AND minutes > 30
            """
            
            result = pd.read_sql(query, self.engine, params={'player_id': player_id})
            
            return result['avg_creativity'].iloc[0] if not result.empty else 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_ict_involvement(self, player_id: int) -> float:
        """Calculate overall ICT index involvement."""
        try:
            query = """
            SELECT AVG(ict_index) as avg_ict
            FROM gameweek_stats
            WHERE player_id = :player_id
            AND gameweek_id > (SELECT MAX(id) - 10 FROM gameweeks WHERE finished = TRUE)
            AND minutes > 30
            """
            
            result = pd.read_sql(query, self.engine, params={'player_id': player_id})
            
            return result['avg_ict'].iloc[0] if not result.empty else 0.0
            
        except Exception:
            return 0.0
    
    def _is_penalty_taker(self, player_id: int) -> int:
        """Determine if player is primary penalty taker (1=yes, 0=no)."""
        try:
            # This would ideally use penalty data, approximating from player analysis
            query = """
            SELECT web_name, team
            FROM players
            WHERE id = :player_id
            """
            
            result = pd.read_sql(query, self.engine, params={'player_id': player_id})
            
            if result.empty:
                return 0
            
            # This would need penalty taker database - placeholder logic
            # In reality, would maintain penalty taker database
            return 0  # Conservative default
            
        except Exception:
            return 0
    
    def _is_free_kick_taker(self, player_id: int) -> int:
        """Determine if player takes free kicks."""
        # Similar to penalty taker logic
        return 0  # Placeholder
    
    def _calculate_shots_per_game(self, player_id: int) -> float:
        """Calculate shots per game (approximated from threat)."""
        try:
            query = """
            SELECT AVG(threat) as avg_threat
            FROM gameweek_stats
            WHERE player_id = :player_id
            AND gameweek_id > (SELECT MAX(id) - 10 FROM gameweeks WHERE finished = TRUE)
            AND minutes > 30
            """
            
            result = pd.read_sql(query, self.engine, params={'player_id': player_id})
            
            if result.empty:
                return 0.0
            
            # Approximate shots from threat index
            threat = result['avg_threat'].iloc[0] or 0
            shots_per_game = threat / 50  # Rough conversion
            
            return shots_per_game
            
        except Exception:
            return 0.0
    
    def _calculate_penalty_conversion(self, player_id: int) -> float:
        """Calculate penalty conversion rate."""
        # Would need penalty data - returning default
        return 0.8  # Typical conversion rate
    
    def _calculate_threat_index(self, player_id: int) -> float:
        """Calculate threat index for forwards."""
        try:
            query = """
            SELECT AVG(threat) as avg_threat
            FROM gameweek_stats
            WHERE player_id = :player_id
            AND gameweek_id > (SELECT MAX(id) - 10 FROM gameweeks WHERE finished = TRUE)
            AND minutes > 30
            """
            
            result = pd.read_sql(query, self.engine, params={'player_id': player_id})
            
            return result['avg_threat'].iloc[0] if not result.empty else 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_minutes_certainty(self, player_id: int) -> float:
        """Calculate how certain the player is to play significant minutes."""
        try:
            query = """
            SELECT 
                AVG(CASE WHEN minutes >= 60 THEN 1 ELSE 0 END) as start_frequency,
                AVG(minutes) as avg_minutes
            FROM gameweek_stats
            WHERE player_id = :player_id
            AND gameweek_id > (SELECT MAX(id) - 10 FROM gameweeks WHERE finished = TRUE)
            """
            
            result = pd.read_sql(query, self.engine, params={'player_id': player_id})
            
            if result.empty:
                return 0.7
            
            start_frequency = result['start_frequency'].iloc[0] or 0
            avg_minutes = result['avg_minutes'].iloc[0] or 0
            
            # Combine start frequency with average minutes
            minutes_certainty = (start_frequency + (avg_minutes / 90)) / 2
            
            return min(minutes_certainty, 1.0)
            
        except Exception:
            return 0.7
    
    def _calculate_rotation_risk(self, player_id: int) -> float:
        """Calculate rotation risk (inverse of minutes certainty)."""
        minutes_certainty = self._calculate_minutes_certainty(player_id)
        return 1.0 - minutes_certainty
    
    # Placeholder methods for additional features - would be expanded based on available data
    
    def _get_goalkeeper_historical_stats(self, player_id: int) -> Dict:
        """Get comprehensive historical stats for goalkeeper."""
        return {
            'saves_per_game_5gw': 3.2,
            'clean_sheet_rate_5gw': 0.4,
            'goals_conceded_per_game': 1.1,
            'save_percentage': 0.7
        }
    
    def _get_team_defensive_metrics(self, team_id: int) -> Dict:
        """Get team defensive strength metrics."""
        return {
            'team_clean_sheet_rate': 0.35,
            'team_goals_conceded_per_game': 1.2,
            'team_defensive_stability': 0.8
        }
    
    def _get_goalkeeper_fixture_features(self, player_id: int, gameweek_id: int) -> Dict:
        """Get goalkeeper-specific fixture features."""
        return {
            'fixture_difficulty_gk': 3,
            'opponent_goals_per_game': 1.5,
            'home_away_factor': 1.1
        }
    
    def _calculate_advanced_gk_features(self, player_id: int) -> Dict:
        """Calculate advanced goalkeeper features."""
        return {
            'distribution_accuracy': 0.8,
            'command_of_area': 0.7,
            'shot_stopping_ability': 0.75
        }
    
    def _get_defender_historical_stats(self, player_id: int) -> Dict:
        """Get historical stats for defenders."""
        return {
            'clean_sheets_5gw': 2,
            'goal_involvements_5gw': 1,
            'aerial_duels_won': 0.6,
            'tackles_per_game': 2.5
        }
    
    def _get_team_context_features(self, team_id: int, gameweek_id: int) -> Dict:
        """Get team context for defenders."""
        return {
            'team_attack_strength': 3.5,
            'team_set_piece_threat': 0.6,
            'team_form': 2.1
        }
    
    def _calculate_advanced_defender_features(self, player_id: int) -> Dict:
        """Advanced defender-specific features."""
        return {
            'aerial_threat': 0.3,
            'crossing_accuracy': 0.25,
            'defensive_actions_per_game': 8.5
        }
    
    def _get_midfielder_historical_stats(self, player_id: int) -> Dict:
        """Historical stats for midfielders."""
        return {
            'key_passes_per_game': 2.1,
            'shots_on_target_per_game': 1.2,
            'dribbles_per_game': 1.8,
            'pass_completion_rate': 0.85
        }
    
    def _get_team_attacking_context(self, team_id: int, gameweek_id: int) -> Dict:
        """Team attacking context for midfielders."""
        return {
            'team_goals_per_game': 1.7,
            'team_possession_rate': 0.58,
            'team_attacking_tempo': 0.7
        }
    
    def _calculate_advanced_midfielder_features(self, player_id: int) -> Dict:
        """Advanced midfielder features."""
        return {
            'progressive_passes': 5.2,
            'final_third_entries': 8.1,
            'box_entries_per_game': 3.4
        }
    
    def _get_forward_historical_stats(self, player_id: int) -> Dict:
        """Historical stats for forwards."""
        return {
            'shots_per_game': 3.2,
            'shots_on_target_rate': 0.4,
            'conversion_rate': 0.15,
            'big_chances_scored': 0.6
        }
    
    def _get_opposition_defensive_analysis(self, player_id: int, gameweek_id: int) -> Dict:
        """Opposition defensive analysis for forwards."""
        return {
            'opponent_goals_conceded_per_game': 1.3,
            'opponent_clean_sheet_rate': 0.3,
            'opponent_defensive_errors': 1.1
        }
    
    def _calculate_advanced_forward_features(self, player_id: int) -> Dict:
        """Advanced forward features."""
        return {
            'penalty_box_touches': 12.5,
            'aerial_duels_success': 0.55,
            'link_up_play_rating': 0.7
        }



