import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sqlalchemy import create_engine
import logging
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

from models.optimization.squad_optimizer import SquadOptimizer
from transfers.transfer_optimizer import TransferOptimizer
from transfers.hit_analyzer import TransferHitAnalyzer

load_dotenv()

logger = logging.getLogger(__name__)

class MultiGameweekPlanner:
    """
    Strategic planning system for optimizing transfers and squad decisions
    across multiple gameweeks considering fixture swings, price changes,
    and chip usage timing.
    """
    
    def __init__(self):
        self.engine = create_engine(os.getenv('DATABASE_URL'))
        self.squad_optimizer = SquadOptimizer()
        self.transfer_optimizer = TransferOptimizer()
        self.hit_analyzer = TransferHitAnalyzer()
        
        # Planning parameters
        self.MAX_PLANNING_HORIZON = 8  # Maximum gameweeks to plan ahead
        self.CHIP_PLANNING_HORIZON = 12  # Extended horizon for chip planning
        
        # Strategic weights
        self.WEIGHTS = {
            'immediate_points': 0.4,
            'fixture_ease': 0.3,
            'price_changes': 0.15,
            'differentials': 0.1,
            'chip_synergy': 0.05
        }
    
    def create_multi_gameweek_plan(self, current_squad: List[Dict], 
                                 current_gw: int, planning_horizon: int,
                                 available_chips: List[str],
                                 strategy_type: str = "balanced") -> Dict:
        """
        Create comprehensive multi-gameweek strategic plan.
        
        Args:
            current_squad: Current 15-player squad
            current_gw: Current gameweek number
            planning_horizon: Number of gameweeks to plan
            available_chips: List of available chips
            strategy_type: Planning strategy (balanced, aggressive, conservative)
            
        Returns:
            Comprehensive strategic plan
        """
        try:
            logger.info(f"Creating {planning_horizon}-gameweek strategic plan...")
            
            end_gw = min(current_gw + planning_horizon - 1, 38)
            gameweeks = list(range(current_gw, end_gw + 1))
            
            # Get multi-gameweek predictions
            predictions_multi_gw = self._get_multi_gameweek_predictions(gameweeks)
            
            # Analyze fixture difficulty patterns
            fixture_analysis = self._analyze_fixture_patterns(gameweeks)
            
            # Plan optimal chip usage timing
            chip_strategy = self._plan_chip_usage(
                available_chips, gameweeks, predictions_multi_gw, fixture_analysis
            )
            
            # Generate transfer timeline
            transfer_timeline = self._create_transfer_timeline(
                current_squad, gameweeks, predictions_multi_gw, 
                fixture_analysis, chip_strategy, strategy_type
            )
            
            # Calculate expected outcomes
            outcome_projections = self._project_outcomes(
                current_squad, transfer_timeline, predictions_multi_gw
            )
            
            # Risk assessment
            risk_assessment = self._assess_strategic_risks(
                transfer_timeline, chip_strategy, gameweeks
            )
            
            # Generate key insights and recommendations
            insights = self._generate_strategic_insights(
                transfer_timeline, chip_strategy, outcome_projections, 
                fixture_analysis, risk_assessment
            )
            
            return {
                'planning_period': {
                    'start_gameweek': current_gw,
                    'end_gameweek': end_gw,
                    'total_gameweeks': len(gameweeks)
                },
                'transfer_timeline': transfer_timeline,
                'chip_strategy': chip_strategy,
                'fixture_analysis': fixture_analysis,
                'outcome_projections': outcome_projections,
                'risk_assessment': risk_assessment,
                'strategic_insights': insights,
                'strategy_type': strategy_type,
                'created_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error creating multi-gameweek plan: {e}")
            raise
    
    def _get_multi_gameweek_predictions(self, gameweeks: List[int]) -> Dict[int, Dict]:
        """Get predictions for multiple gameweeks."""
        try:
            # This would typically call the predictor for each gameweek
            # For now, returning placeholder structure
            
            predictions_multi_gw = {}
            
            for gw in gameweeks:
                # Placeholder predictions - would call actual predictor
                predictions_multi_gw[gw] = {
                    # player_id: {'points': float, 'confidence': float}
                    # This would be populated by actual predictions
                }
            
            return predictions_multi_gw
            
        except Exception as e:
            logger.error(f"Error getting multi-gameweek predictions: {e}")
            return {}
    
    def _analyze_fixture_patterns(self, gameweeks: List[int]) -> Dict:
        """Analyze fixture difficulty patterns and identify key swing periods."""
        try:
            logger.info("Analyzing fixture patterns...")
            
            # Get fixture data for the planning period
            query = """
            SELECT 
                f.gameweek_id,
                f.team_h,
                f.team_a,
                f.team_h_difficulty,
                f.team_a_difficulty,
                th.name as home_team_name,
                ta.name as away_team_name
            FROM fixtures f
            JOIN teams th ON f.team_h = th.id
            JOIN teams ta ON f.team_a = ta.id
            WHERE f.gameweek_id IN :gameweeks
            ORDER BY f.gameweek_id, f.team_h
            """
            
            fixtures_df = pd.read_sql(
                query, self.engine, 
                params={'gameweeks': tuple(gameweeks)}
            )
            
            if fixtures_df.empty:
                return {'fixture_swings': [], 'key_periods': []}
            
            # Analyze team fixture difficulty patterns
            team_difficulty_patterns = self._calculate_team_difficulty_patterns(fixtures_df)
            
            # Identify fixture swings
            fixture_swings = self._identify_fixture_swings(team_difficulty_patterns, gameweeks)
            
            # Identify key periods for transfers
            key_periods = self._identify_key_transfer_periods(fixture_swings, gameweeks)
            
            # Calculate double gameweeks and blank gameweeks
            special_gameweeks = self._identify_special_gameweeks(fixtures_df, gameweeks)
            
            return {
                'team_difficulty_patterns': team_difficulty_patterns,
                'fixture_swings': fixture_swings,
                'key_periods': key_periods,
                'special_gameweeks': special_gameweeks,
                'average_difficulty_by_gw': self._calculate_average_difficulty_by_gw(fixtures_df)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing fixture patterns: {e}")
            return {}
    
    def _calculate_team_difficulty_patterns(self, fixtures_df: pd.DataFrame) -> Dict:
        """Calculate difficulty patterns for each team."""
        team_patterns = {}
        
        for team_id in pd.concat([fixtures_df['team_h'], fixtures_df['team_a']]).unique():
            team_fixtures = []
            
            # Get home fixtures
            home_fixtures = fixtures_df[fixtures_df['team_h'] == team_id]
            for _, fixture in home_fixtures.iterrows():
                team_fixtures.append({
                    'gameweek': fixture['gameweek_id'],
                    'difficulty': fixture['team_h_difficulty'],
                    'opponent': fixture['away_team_name'],
                    'is_home': True
                })
            
            # Get away fixtures
            away_fixtures = fixtures_df[fixtures_df['team_a'] == team_id]
            for _, fixture in away_fixtures.iterrows():
                team_fixtures.append({
                    'gameweek': fixture['gameweek_id'],
                    'difficulty': fixture['team_a_difficulty'],
                    'opponent': fixture['home_team_name'],
                    'is_home': False
                })
            
            # Sort by gameweek
            team_fixtures.sort(key=lambda x: x['gameweek'])
            
            # Calculate rolling difficulty average
            difficulties = [f['difficulty'] for f in team_fixtures]
            rolling_avg = pd.Series(difficulties).rolling(window=3, min_periods=1).mean().tolist()
            
            team_patterns[team_id] = {
                'fixtures': team_fixtures,
                'difficulties': difficulties,
                'rolling_avg_difficulty': rolling_avg,
                'best_period': self._find_best_fixture_period(difficulties),
                'worst_period': self._find_worst_fixture_period(difficulties)
            }
        
        return team_patterns
    
    def _identify_fixture_swings(self, team_patterns: Dict, gameweeks: List[int]) -> List[Dict]:
        """Identify significant fixture difficulty swings."""
        fixture_swings = []
        
        for team_id, pattern in team_patterns.items():
            difficulties = pattern['difficulties']
            
            if len(difficulties) < 3:
                continue
            
            # Look for significant changes in rolling difficulty
            for i in range(len(difficulties) - 2):
                current_avg = sum(difficulties[i:i+3]) / 3
                
                if i + 3 < len(difficulties):
                    next_avg = sum(difficulties[i+1:i+4]) / 3
                    
                    # Identify positive swings (easier fixtures)
                    if current_avg - next_avg >= 1.0:  # Significant improvement
                        swing_start_gw = pattern['fixtures'][i+1]['gameweek']
                        
                        fixture_swings.append({
                            'team_id': team_id,
                            'type': 'positive',
                            'start_gameweek': swing_start_gw,
                            'difficulty_improvement': current_avg - next_avg,
                            'duration': 3,
                            'impact': 'high' if current_avg - next_avg >= 1.5 else 'medium'
                        })
        
        # Sort by impact and start gameweek
        fixture_swings.sort(key=lambda x: (x['start_gameweek'], -x['difficulty_improvement']))
        
        return fixture_swings
    
    def _identify_key_transfer_periods(self, fixture_swings: List[Dict], gameweeks: List[int]) -> List[Dict]:
        """Identify optimal periods for making transfers based on fixture swings."""
        key_periods = []
        
        # Group swings by gameweek
        swings_by_gw = {}
        for swing in fixture_swings:
            gw = swing['start_gameweek']
            if gw not in swings_by_gw:
                swings_by_gw[gw] = []
            swings_by_gw[gw].append(swing)
        
        # Identify periods with multiple positive swings
        for gw, swings in swings_by_gw.items():
            if len(swings) >= 2:  # Multiple teams with easier fixtures
                high_impact_swings = [s for s in swings if s['impact'] == 'high']
                
                key_periods.append({
                    'gameweek': gw,
                    'type': 'fixture_swing',
                    'swing_count': len(swings),
                    'high_impact_count': len(high_impact_swings),
                    'priority': 'high' if len(high_impact_swings) >= 2 else 'medium',
                    'recommended_action': 'Target players from teams with easier fixtures'
                })
        
        # Add other strategic periods
        if gameweeks[0] in [1, 2]:  # Season start
            key_periods.append({
                'gameweek': gameweeks[0],
                'type': 'season_start',
                'priority': 'high',
                'recommended_action': 'Establish premium core players'
            })
        
        return key_periods
    
    def _plan_chip_usage(self, available_chips: List[str], gameweeks: List[int],
                        predictions_multi_gw: Dict, fixture_analysis: Dict) -> Dict:
        """Plan optimal timing for chip usage."""
        try:
            chip_strategy = {
                'planned_chips': [],
                'chip_gameweeks': {},
                'synergy_opportunities': []
            }
            
            if not available_chips:
                return chip_strategy
            
            # Wildcard planning
            if 'Wildcard' in available_chips:
                wildcard_timing = self._plan_wildcard_timing(
                    gameweeks, fixture_analysis, predictions_multi_gw
                )
                chip_strategy['planned_chips'].append(wildcard_timing)
                chip_strategy['chip_gameweeks']['Wildcard'] = wildcard_timing['optimal_gameweek']
            
            # Triple Captain planning
            if 'Triple Captain' in available_chips:
                tc_timing = self._plan_triple_captain_timing(
                    gameweeks, predictions_multi_gw, fixture_analysis
                )
                chip_strategy['planned_chips'].append(tc_timing)
                chip_strategy['chip_gameweeks']['Triple Captain'] = tc_timing['optimal_gameweek']
            
            # Bench Boost planning
            if 'Bench Boost' in available_chips:
                bb_timing = self._plan_bench_boost_timing(
                    gameweeks, predictions_multi_gw, fixture_analysis
                )
                chip_strategy['planned_chips'].append(bb_timing)
                chip_strategy['chip_gameweeks']['Bench Boost'] = bb_timing['optimal_gameweek']
            
            # Free Hit planning
            if 'Free Hit' in available_chips:
                fh_timing = self._plan_free_hit_timing(
                    gameweeks, fixture_analysis
                )
                chip_strategy['planned_chips'].append(fh_timing)
                chip_strategy['chip_gameweeks']['Free Hit'] = fh_timing['optimal_gameweek']
            
            # Identify synergy opportunities
            chip_strategy['synergy_opportunities'] = self._identify_chip_synergies(
                chip_strategy['planned_chips'], fixture_analysis
            )
            
            return chip_strategy
            
        except Exception as e:
            logger.error(f"Error planning chip usage: {e}")
            return {}
    
    def _plan_wildcard_timing(self, gameweeks: List[int], fixture_analysis: Dict,
                            predictions_multi_gw: Dict) -> Dict:
        """Plan optimal wildcard timing."""
        # Look for periods with multiple fixture swings
        optimal_gw = gameweeks[0]  # Default to first gameweek
        max_benefit = 0
        
        for swing in fixture_analysis.get('fixture_swings', []):
            if swing['type'] == 'positive' and swing['start_gameweek'] in gameweeks:
                benefit = swing['difficulty_improvement'] * 5  # Weight fixture benefit
                if benefit > max_benefit:
                    max_benefit = benefit
                    optimal_gw = swing['start_gameweek']
        
        return {
            'chip': 'Wildcard',
            'optimal_gameweek': optimal_gw,
            'reasoning': 'Maximize fixture swing benefit',
            'expected_benefit': max_benefit,
            'preparation_gameweeks': [optimal_gw - 1] if optimal_gw > gameweeks[0] else []
        }
    
    def _plan_triple_captain_timing(self, gameweeks: List[int], predictions_multi_gw: Dict,
                                  fixture_analysis: Dict) -> Dict:
        """Plan optimal triple captain timing."""
        # Look for gameweeks with high predicted captain scores
        optimal_gw = gameweeks[0]
        max_captain_score = 0
        
        # This would analyze predictions for top captain candidates
        # For now, using placeholder logic
        
        for gw in gameweeks:
            # Placeholder: look for easy fixture gameweeks
            avg_difficulty = fixture_analysis.get('average_difficulty_by_gw', {}).get(gw, 3)
            captain_score = (5 - avg_difficulty) * 4  # Simplified scoring
            
            if captain_score > max_captain_score:
                max_captain_score = captain_score
                optimal_gw = gw
        
        return {
            'chip': 'Triple Captain',
            'optimal_gameweek': optimal_gw,
            'reasoning': 'Target gameweek with easiest fixtures for premium players',
            'expected_benefit': max_captain_score,
            'recommended_captain': 'Premium forward/midfielder with easy fixture'
        }
    
    def _plan_bench_boost_timing(self, gameweeks: List[int], predictions_multi_gw: Dict,
                               fixture_analysis: Dict) -> Dict:
        """Plan optimal bench boost timing."""
        # Look for double gameweeks or periods with many good fixtures
        optimal_gw = gameweeks[-1]  # Default to later in planning period
        
        special_gws = fixture_analysis.get('special_gameweeks', {})
        
        # Prefer double gameweeks if available
        for gw in gameweeks:
            if special_gws.get(gw, {}).get('type') == 'double':
                optimal_gw = gw
                break
        
        return {
            'chip': 'Bench Boost',
            'optimal_gameweek': optimal_gw,
            'reasoning': 'Target double gameweek or period with good bench fixtures',
            'expected_benefit': 15,  # Typical bench boost benefit
            'preparation_required': 'Build strong bench 1-2 gameweeks prior'
        }
    
    def _plan_free_hit_timing(self, gameweeks: List[int], fixture_analysis: Dict) -> Dict:
        """Plan optimal free hit timing."""
        # Look for blank gameweeks or exceptional fixture swings
        optimal_gw = gameweeks[0]
        
        special_gws = fixture_analysis.get('special_gameweeks', {})
        
        # Prefer blank gameweeks
        for gw in gameweeks:
            if special_gws.get(gw, {}).get('type') == 'blank':
                optimal_gw = gw
                break
        
        return {
            'chip': 'Free Hit',
            'optimal_gameweek': optimal_gw,
            'reasoning': 'Target blank gameweek or exceptional fixture set',
            'expected_benefit': 20,  # Typical free hit benefit
            'strategy': 'Load up on players with best fixtures regardless of ownership'
        }
    
    def _create_transfer_timeline(self, current_squad: List[Dict], gameweeks: List[int],
                                predictions_multi_gw: Dict, fixture_analysis: Dict,
                                chip_strategy: Dict, strategy_type: str) -> List[Dict]:
        """Create detailed transfer timeline."""
        try:
            transfer_timeline = []
            working_squad = current_squad.copy()
            
            for i, gw in enumerate(gameweeks):
                gw_plan = {
                    'gameweek': gw,
                    'transfers': [],
                    'chip_usage': None,
                    'reasoning': [],
                    'squad_changes': [],
                    'expected_points_gain': 0,
                    'risk_level': 'medium'
                }
                
                # Check for chip usage
                for chip_data in chip_strategy.get('planned_chips', []):
                    if chip_data['optimal_gameweek'] == gw:
                        gw_plan['chip_usage'] = chip_data['chip']
                        gw_plan['reasoning'].append(f"Use {chip_data['chip']}: {chip_data['reasoning']}")
                
                # Plan transfers based on strategy and analysis
                if gw_plan['chip_usage'] == 'Wildcard':
                    # Wildcard gameweek - comprehensive squad overhaul
                    wildcard_plan = self._plan_wildcard_transfers(
                        working_squad, gw, predictions_multi_gw.get(gw, {}),
                        fixture_analysis, strategy_type
                    )
                    gw_plan.update(wildcard_plan)
                    working_squad = wildcard_plan['new_squad']
                
                elif gw_plan['chip_usage'] == 'Free Hit':
                    # Free hit - optimal squad for one week only
                    free_hit_plan = self._plan_free_hit_squad(
                        gw, predictions_multi_gw.get(gw, {}), fixture_analysis
                    )
                    gw_plan.update(free_hit_plan)
                    # Squad reverts after free hit
                
                else:
                    # Regular transfers
                    transfer_plan = self._plan_regular_transfers(
                        working_squad, gw, predictions_multi_gw.get(gw, {}),
                        fixture_analysis, strategy_type
                    )
                    gw_plan.update(transfer_plan)
                    
                    # Update working squad
                    for transfer in transfer_plan.get('transfers', []):
                        # Remove transferred out players
                        working_squad = [p for p in working_squad if p['id'] not in [po['id'] for po in transfer.get('players_out', [])]]
                        # Add transferred in players
                        working_squad.extend(transfer.get('players_in', []))
                
                transfer_timeline.append(gw_plan)
            
            return transfer_timeline
            
        except Exception as e:
            logger.error(f"Error creating transfer timeline: {e}")
            return []
    
    # Additional helper methods would be implemented for:
    # - _plan_wildcard_transfers
    # - _plan_free_hit_squad  
    # - _plan_regular_transfers
    # - _project_outcomes
    # - _assess_strategic_risks
    # - _generate_strategic_insights
    # etc.
    
    def get_optimal_transfer_timing(self, transfer_options: List[Dict],
                                  gameweeks: List[int],
                                  predictions_multi_gw: Dict) -> Dict:
        """Determine optimal timing for specific transfers across multiple gameweeks."""
        try:
            timing_analysis = {}
            
            for option in transfer_options:
                option_analysis = {}
                
                for gw in gameweeks:
                    # Analyze transfer value for this specific gameweek
                    gw_predictions = predictions_multi_gw.get(gw, {})
                    
                    # Calculate immediate and future value
                    immediate_value = self._calculate_immediate_transfer_value(option, gw_predictions)
                    future_value = self._calculate_future_transfer_value(option, gw, predictions_multi_gw)
                    
                    total_value = immediate_value + future_value
                    
                    option_analysis[gw] = {
                        'immediate_value': immediate_value,
                        'future_value': future_value,
                        'total_value': total_value,
                        'timing_score': self._calculate_timing_score(option, gw, predictions_multi_gw)
                    }
                
                # Find optimal gameweek for this transfer
                optimal_gw = max(option_analysis.keys(), key=lambda gw: option_analysis[gw]['total_value'])
                
                timing_analysis[f"transfer_{len(timing_analysis)}"] = {
                    'transfer_option': option,
                    'gameweek_analysis': option_analysis,
                    'optimal_gameweek': optimal_gw,
                    'optimal_value': option_analysis[optimal_gw]['total_value'],
                    'timing_confidence': self._calculate_timing_confidence(option_analysis)
                }
            
            return timing_analysis
            
        except Exception as e:
            logger.error(f"Error determining optimal transfer timing: {e}")
            return {}
    
    # Placeholder helper methods (would be fully implemented)
    def _find_best_fixture_period(self, difficulties: List[int]) -> Dict:
        return {'start_index': 0, 'duration': 3, 'avg_difficulty': 2.0}
    
    def _find_worst_fixture_period(self, difficulties: List[int]) -> Dict:
        return {'start_index': 3, 'duration': 3, 'avg_difficulty': 4.0}
    
    def _identify_special_gameweeks(self, fixtures_df: pd.DataFrame, gameweeks: List[int]) -> Dict:
        return {}  # Would identify double/blank gameweeks
    
    def _calculate_average_difficulty_by_gw(self, fixtures_df: pd.DataFrame) -> Dict:
        return {}  # Would calculate average difficulty per gameweek
    
    def _identify_chip_synergies(self, planned_chips: List[Dict], fixture_analysis: Dict) -> List[Dict]:
        return []  # Would identify opportunities to combine chips effectively
    
    def _calculate_immediate_transfer_value(self, option: Dict, gw_predictions: Dict) -> float:
        return 5.0  # Placeholder
    
    def _calculate_future_transfer_value(self, option: Dict, gw: int, predictions_multi_gw: Dict) -> float:
        return 10.0  # Placeholder
    
    def _calculate_timing_score(self, option: Dict, gw: int, predictions_multi_gw: Dict) -> float:
        return 0.8  # Placeholder
    
    def _calculate_timing_confidence(self, option_analysis: Dict) -> float:
        return 0.75  # Placeholder
