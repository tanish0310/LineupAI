import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sqlalchemy import create_engine
import logging
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

class TransferReasoning:
    """
    Advanced reasoning engine that generates human-readable explanations
    for transfer recommendations with detailed analysis and context.
    """
    
    def __init__(self):
        self.engine = create_engine(os.getenv('DATABASE_URL'))
        
        # Reasoning templates and thresholds
        self.STRONG_IMPROVEMENT_THRESHOLD = 3.0
        self.GOOD_IMPROVEMENT_THRESHOLD = 1.5
        self.FORM_DIFFERENCE_THRESHOLD = 1.0
        self.PRICE_CHANGE_THRESHOLD = 0.3
        
        # Fixture difficulty ratings
        self.FIXTURE_DIFFICULTY = {
            1: 'Very Easy',
            2: 'Easy', 
            3: 'Moderate',
            4: 'Hard',
            5: 'Very Hard'
        }
    
    def generate_transfer_explanation(self, player_out: Dict, player_in: Dict, 
                                    predictions: Dict[int, Dict], fixtures: Dict = None,
                                    gameweek_id: int = None) -> Dict:
        """
        Generate comprehensive explanation for a single transfer recommendation.
        
        Args:
            player_out: Player being transferred out
            player_in: Player being transferred in
            predictions: Player predictions
            fixtures: Fixture data for analysis
            gameweek_id: Current gameweek for context
            
        Returns:
            Detailed explanation with reasoning categories
        """
        try:
            logger.info(f"Generating transfer explanation: {player_out['name']} → {player_in['name']}")
            
            explanation = {
                'transfer_summary': f"{player_out['name']} → {player_in['name']}",
                'primary_reasons': [],
                'supporting_factors': [],
                'risk_factors': [],
                'confidence_level': 'Medium',
                'recommendation_strength': 'Consider',
                'detailed_analysis': {},
                'expected_impact': {}
            }
            
            # Calculate basic metrics
            points_out = predictions.get(player_out['id'], {}).get('points', 0)
            points_in = predictions.get(player_in['id'], {}).get('points', 0)
            points_improvement = points_in - points_out
            
            # Performance comparison analysis
            performance_analysis = self._analyze_performance_difference(
                player_out, player_in, points_improvement, predictions
            )
            explanation['primary_reasons'].extend(performance_analysis['primary_reasons'])
            explanation['detailed_analysis']['performance'] = performance_analysis['details']
            
            # Form analysis
            form_analysis = self._analyze_form_difference(player_out, player_in)
            if form_analysis['significant']:
                explanation['supporting_factors'].extend(form_analysis['reasons'])
            explanation['detailed_analysis']['form'] = form_analysis['details']
            
            # Fixture analysis
            if fixtures and gameweek_id:
                fixture_analysis = self._analyze_fixture_difference(
                    player_out, player_in, fixtures, gameweek_id
                )
                if fixture_analysis['significant']:
                    explanation['supporting_factors'].extend(fixture_analysis['reasons'])
                explanation['detailed_analysis']['fixtures'] = fixture_analysis['details']
            
            # Price and value analysis
            price_analysis = self._analyze_price_and_value(player_out, player_in, points_improvement, predictions)
            if price_analysis['significant']:
                explanation['supporting_factors'].extend(price_analysis['reasons'])
            explanation['detailed_analysis']['pricing'] = price_analysis['details']
            
            # Injury and availability analysis
            availability_analysis = self._analyze_availability_risks(player_out, player_in)
            if availability_analysis['concerns']:
                explanation['risk_factors'].extend(availability_analysis['risks'])
            elif availability_analysis['improvements']:
                explanation['supporting_factors'].extend(availability_analysis['benefits'])
            explanation['detailed_analysis']['availability'] = availability_analysis['details']
            
            # Team context analysis
            team_analysis = self._analyze_team_context(player_out, player_in)
            if team_analysis['significant']:
                explanation['supporting_factors'].extend(team_analysis['reasons'])
            explanation['detailed_analysis']['team_context'] = team_analysis['details']
            
            # Calculate overall confidence and recommendation strength
            confidence_metrics = self._calculate_transfer_confidence(
                points_improvement, performance_analysis, form_analysis, 
                availability_analysis, price_analysis
            )
            explanation['confidence_level'] = confidence_metrics['level']
            explanation['recommendation_strength'] = confidence_metrics['strength']
            
            # Expected impact summary
            explanation['expected_impact'] = self._calculate_expected_impact(
                player_out, player_in, points_improvement, confidence_metrics
            )
            
            # Generate summary explanation
            explanation['summary_explanation'] = self._generate_summary_explanation(
                explanation, points_improvement
            )
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error generating transfer explanation: {e}")
            return self._create_error_explanation(player_out, player_in)
    
    def _analyze_performance_difference(self, player_out: Dict, player_in: Dict, 
                                      points_improvement: float, predictions: Dict) -> Dict:
        """Analyze the core performance difference between players."""
        try:
            analysis = {
                'primary_reasons': [],
                'details': {
                    'points_improvement': points_improvement,
                    'player_out_predicted': predictions.get(player_out['id'], {}).get('points', 0),
                    'player_in_predicted': predictions.get(player_in['id'], {}).get('points', 0)
                }
            }
            
            if points_improvement >= self.STRONG_IMPROVEMENT_THRESHOLD:
                analysis['primary_reasons'].append(
                    f"Strong predicted improvement: +{points_improvement:.1f} points per gameweek"
                )
            elif points_improvement >= self.GOOD_IMPROVEMENT_THRESHOLD:
                analysis['primary_reasons'].append(
                    f"Good predicted improvement: +{points_improvement:.1f} points per gameweek"
                )
            elif points_improvement > 0:
                analysis['primary_reasons'].append(
                    f"Modest improvement expected: +{points_improvement:.1f} points per gameweek"
                )
            else:
                analysis['primary_reasons'].append(
                    "Strategic move (no immediate point improvement expected)"
                )
            
            # Add position-specific performance context
            position_context = self._get_position_performance_context(
                player_out, player_in, predictions
            )
            if position_context:
                analysis['primary_reasons'].append(position_context)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in performance analysis: {e}")
            return {'primary_reasons': [], 'details': {}}
    
    def _get_position_performance_context(self, player_out: Dict, player_in: Dict, 
                                        predictions: Dict) -> Optional[str]:
        """Get position-specific performance context."""
        position = player_out.get('position', 0)
        
        if position == 1:  # Goalkeeper
            return "Goalkeeper upgrade focusing on clean sheet potential and save points"
        elif position == 2:  # Defender
            clean_sheets_in = predictions.get(player_in['id'], {}).get('clean_sheet_probability', 0)
            clean_sheets_out = predictions.get(player_out['id'], {}).get('clean_sheet_probability', 0)
            if clean_sheets_in > clean_sheets_out + 0.1:
                return "Defensive upgrade with better clean sheet prospects"
            return "Defensive change targeting attacking returns and bonus points"
        elif position == 3:  # Midfielder
            return "Midfield upgrade focusing on goals, assists, and creative involvement"
        elif position == 4:  # Forward
            return "Forward upgrade targeting increased goal threat and attacking returns"
        
        return None
    
    def _analyze_form_difference(self, player_out: Dict, player_in: Dict) -> Dict:
        """Analyze form difference between players."""
        try:
            form_out = player_out.get('form', 0)
            form_in = player_in.get('form', 0)
            form_difference = form_in - form_out
            
            analysis = {
                'significant': abs(form_difference) >= self.FORM_DIFFERENCE_THRESHOLD,
                'reasons': [],
                'details': {
                    'form_out': form_out,
                    'form_in': form_in,
                    'difference': form_difference
                }
            }
            
            if form_difference >= self.FORM_DIFFERENCE_THRESHOLD:
                if form_in >= 4.0:
                    analysis['reasons'].append(f"{player_in['name']} is in excellent recent form ({form_in:.1f})")
                else:
                    analysis['reasons'].append(f"{player_in['name']} has superior recent form ({form_in:.1f} vs {form_out:.1f})")
            
            if form_out < 2.0:
                analysis['reasons'].append(f"{player_out['name']} has been struggling for form ({form_out:.1f})")
                analysis['significant'] = True
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in form analysis: {e}")
            return {'significant': False, 'reasons': [], 'details': {}}
    
    def _analyze_fixture_difference(self, player_out: Dict, player_in: Dict, 
                                  fixtures: Dict, gameweek_id: int) -> Dict:
        """Analyze upcoming fixture difficulty for both players."""
        try:
            analysis = {
                'significant': False,
                'reasons': [],
                'details': {}
            }
            
            # Get next 3-5 fixtures for both players
            fixtures_out = self._get_upcoming_fixtures(player_out['team'], fixtures, gameweek_id, 4)
            fixtures_in = self._get_upcoming_fixtures(player_in['team'], fixtures, gameweek_id, 4)
            
            if not fixtures_out or not fixtures_in:
                return analysis
            
            # Calculate average difficulty
            avg_difficulty_out = sum([f['difficulty'] for f in fixtures_out]) / len(fixtures_out)
            avg_difficulty_in = sum([f['difficulty'] for f in fixtures_in]) / len(fixtures_in)
            
            difficulty_improvement = avg_difficulty_out - avg_difficulty_in
            
            analysis['details'] = {
                'avg_difficulty_out': avg_difficulty_out,
                'avg_difficulty_in': avg_difficulty_in,
                'improvement': difficulty_improvement,
                'fixtures_out': fixtures_out,
                'fixtures_in': fixtures_in
            }
            
            if difficulty_improvement >= 0.75:  # Significantly easier fixtures
                analysis['significant'] = True
                analysis['reasons'].append(
                    f"{player_in['name']} has much easier upcoming fixtures "
                    f"(avg difficulty {avg_difficulty_in:.1f} vs {avg_difficulty_out:.1f})"
                )
                
                # Highlight specific good fixtures
                easy_fixtures = [f for f in fixtures_in if f['difficulty'] <= 2]
                if easy_fixtures:
                    opponents = [f['opponent'] for f in easy_fixtures[:2]]
                    analysis['reasons'].append(
                        f"Including favorable matchups vs {', '.join(opponents)}"
                    )
            
            elif difficulty_improvement <= -0.75:  # Much harder fixtures
                analysis['significant'] = True
                analysis['reasons'].append(
                    f"Warning: {player_in['name']} faces harder upcoming fixtures "
                    f"(avg difficulty {avg_difficulty_in:.1f} vs {avg_difficulty_out:.1f})"
                )
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in fixture analysis: {e}")
            return {'significant': False, 'reasons': [], 'details': {}}
    
    def _get_upcoming_fixtures(self, team_id: int, fixtures: Dict, 
                         gameweek_id: int, num_fixtures: int) -> List[Dict]:
        """Get upcoming fixtures for a team."""
        try:
            query = """
            SELECT 
                f.gameweek_id,
                CASE 
                    WHEN f.team_h = %s THEN f.team_a
                    ELSE f.team_h
                END as opponent_team,
                CASE 
                    WHEN f.team_h = %s THEN f.team_h_difficulty
                    ELSE f.team_a_difficulty
                END as difficulty,
                CASE 
                    WHEN f.team_h = %s THEN TRUE
                    ELSE FALSE
                END as is_home,
                t.name as opponent
            FROM fixtures f
            JOIN teams t ON (
                CASE 
                    WHEN f.team_h = %s THEN f.team_a
                    ELSE f.team_h
                END = t.id
            )
            WHERE (f.team_h = %s OR f.team_a = %s)
            AND f.gameweek_id >= %s
            AND f.gameweek_id < %s + %s
            ORDER BY f.gameweek_id
            LIMIT %s
            """
        
            fixtures_df = pd.read_sql(
                query, 
                self.engine, 
                params=[team_id, team_id, team_id, team_id, team_id, team_id, 
                    gameweek_id, gameweek_id, num_fixtures, num_fixtures]
            )
        
            return fixtures_df.to_dict('records')
        
        except Exception as e:
            logger.error(f"Error getting fixtures for team {team_id}: {e}")
            return []

    
    def _analyze_price_and_value(self, player_out, player_in, points_improvement, predictions):

        """Analyze price change and value implications."""
        try:
            cost_out = player_out.get('selling_price', player_out.get('cost', 0))
            cost_in = player_in.get('cost', 0)
            price_change = (cost_in - cost_out) / 10  # Convert to millions
            
            analysis = {
                'significant': abs(price_change) >= self.PRICE_CHANGE_THRESHOLD,
                'reasons': [],
                'details': {
                    'price_change': price_change,
                    'cost_out': cost_out / 10,
                    'cost_in': cost_in / 10,
                    'value_improvement': points_improvement / max(cost_in / 10, 0.1)
                }
            }
            
            if price_change <= -0.5:  # Significant money saved
                analysis['reasons'].append(
                    f"Frees up £{abs(price_change):.1f}m in squad value"
                )
                analysis['significant'] = True
            elif price_change >= 0.5:  # Significant investment
                if points_improvement > 0:
                    value_ratio = points_improvement / price_change
                    if value_ratio >= 2.0:
                        analysis['reasons'].append(
                            f"Excellent value despite £{price_change:.1f}m increase ({value_ratio:.1f} points per £1m)"
                        )
                    else:
                        analysis['reasons'].append(
                            f"Premium investment of £{price_change:.1f}m justified by performance upgrade"
                        )
                else:
                    analysis['reasons'].append(
                        f"Warning: £{price_change:.1f}m investment without immediate returns"
                    )
                analysis['significant'] = True
            
            # Value per million analysis
            points_per_million_out = (predictions.get(player_out['id'], {}).get('points', 0) / 
                                    max(cost_out / 10, 0.1))
            points_per_million_in = (predictions.get(player_in['id'], {}).get('points', 0) / 
                                   max(cost_in / 10, 0.1))
            
            if points_per_million_in > points_per_million_out * 1.2:
                analysis['reasons'].append(
                    f"Superior value: {points_per_million_in:.1f} vs {points_per_million_out:.1f} points per £1m"
                )
                analysis['significant'] = True
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in price analysis: {e}")
            return {'significant': False, 'reasons': [], 'details': {}}
    
    def _analyze_availability_risks(self, player_out: Dict, player_in: Dict) -> Dict:
        """Analyze injury and availability risks."""
        try:
            analysis = {
                'concerns': False,
                'improvements': False,
                'risks': [],
                'benefits': [],
                'details': {}
            }
            
            # Get latest injury/availability data
            availability_out = self._get_player_availability(player_out['id'])
            availability_in = self._get_player_availability(player_in['id'])
            
            analysis['details'] = {
                'availability_out': availability_out,
                'availability_in': availability_in
            }
            
            # Check if player_out has injury concerns
            if availability_out.get('injury_risk', 0) > 0.3:
                analysis['improvements'] = True
                analysis['benefits'].append(
                    f"Removes injury concern: {player_out['name']} {availability_out.get('status_description', '')}"
                )
            
            # Check if player_in has availability issues
            if availability_in.get('injury_risk', 0) > 0.3:
                analysis['concerns'] = True
                analysis['risks'].append(
                    f"Injury risk: {player_in['name']} {availability_in.get('status_description', '')}"
                )
            
            # Check rotation risk
            if availability_in.get('rotation_risk', 0) > 0.4:
                analysis['concerns'] = True
                analysis['risks'].append(
                    f"Rotation risk: {player_in['name']} may not start consistently"
                )
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in availability analysis: {e}")
            return {'concerns': False, 'improvements': False, 'risks': [], 'benefits': [], 'details': {}}
    
    def _get_player_availability(self, player_id: int) -> Dict:
        """Get latest availability information for a player."""
        try:
            query = """
            SELECT 
                p.chance_of_playing_this_round,
                p.chance_of_playing_next_round,
                p.news,
                COALESCE(tn.injury_status, 'available') as injury_status,
                COALESCE(tn.confidence_score, 1.0) as confidence
            FROM players p
            LEFT JOIN (
                SELECT DISTINCT ON (player_id) 
                    player_id, injury_status, confidence_score
                FROM team_news 
                ORDER BY player_id, created_at DESC
            ) tn ON p.id = tn.player_id
            WHERE p.id = %s
            """
        
            result = pd.read_sql(query, self.engine, params=[player_id])
        
            if result.empty:
                return {'injury_risk': 0, 'rotation_risk': 0.3, 'status_description': 'Unknown'}
            
            row = result.iloc[0]
            
            # Calculate injury risk
            injury_risk = 0
            if row['injury_status'] in ['injured', 'doubtful']:
                injury_risk = 0.7
            elif row['chance_of_playing_this_round'] and row['chance_of_playing_this_round'] < 75:
                injury_risk = 0.5
            
            # Estimate rotation risk (would need more sophisticated model)
            rotation_risk = 0.3  # Default assumption
            
            status_desc = ''
            if row['injury_status'] != 'available':
                status_desc = f"({row['injury_status']})"
            if row['news'] and len(row['news']) > 0:
                status_desc += f" - {row['news'][:50]}..."
            
            return {
                'injury_risk': injury_risk,
                'rotation_risk': rotation_risk,
                'status_description': status_desc,
                'chance_playing': row['chance_of_playing_this_round']
            }
            
        except Exception as e:
            logger.error(f"Error getting availability for player {player_id}: {e}")
            return {'injury_risk': 0, 'rotation_risk': 0.3, 'status_description': 'Unknown'}

    
    def _analyze_team_context(self, player_out: Dict, player_in: Dict) -> Dict:
        """Analyze team context and tactical situation."""
        try:
            analysis = {
                'significant': False,
                'reasons': [],
                'details': {}
            }
            
            # Get team performance context
            team_out_performance = self._get_team_performance(player_out.get('team', 1))
            team_in_performance = self._get_team_performance(player_in.get('team', 1))

            
            analysis['details'] = {
                'team_out_performance': team_out_performance,
                'team_in_performance': team_in_performance
            }
            
            # Compare team attacking/defensive strength
            if player_out.get('position', 0) in [3, 4]:  # Attacking players
                attack_improvement = (team_in_performance.get('attack_strength', 3) - 
                                    team_out_performance.get('attack_strength', 3))
                
                if attack_improvement >= 1:
                    analysis['significant'] = True
                    analysis['reasons'].append(
                        f"Joining stronger attacking team (strength {team_in_performance.get('attack_strength', 3)} vs {team_out_performance.get('attack_strength', 3)})"
                    )
            
            elif player_out.get('position', 0) in [1, 2]:  # Defensive players
                defense_improvement = (team_in_performance.get('defense_strength', 3) - 
                                     team_out_performance.get('defense_strength', 3))
                
                if defense_improvement >= 1:
                    analysis['significant'] = True
                    analysis['reasons'].append(
                        f"Joining stronger defensive team (strength {team_in_performance.get('defense_strength', 3)} vs {team_out_performance.get('defense_strength', 3)})"
                    )
            
            # Team form comparison
            form_improvement = (team_in_performance.get('recent_form', 0) - 
                              team_out_performance.get('recent_form', 0))
            
            if form_improvement >= 1.0:
                analysis['significant'] = True
                analysis['reasons'].append(
                    f"Moving to team in better form ({team_in_performance.get('recent_form', 0):.1f} vs {team_out_performance.get('recent_form', 0):.1f} points per game)"
                )
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in team context analysis: {e}")
            return {'significant': False, 'reasons': [], 'details': {}}
    
    def _get_team_performance(self, team_id: int) -> Dict:
        """Get team performance metrics."""
        try:
            query = """
            SELECT 
                strength_attack_home + strength_attack_away as total_attack,
                strength_defence_home + strength_defence_away as total_defense,
                (strength_attack_home + strength_attack_away + 
                strength_defence_home + strength_defence_away) / 4 as overall_strength
            FROM teams
            WHERE id = %s
            """
        
            result = pd.read_sql(query, self.engine, params=[team_id])
        
        # ... rest of your method remains the same

            
            if result.empty:
                return {'attack_strength': 3, 'defense_strength': 3, 'recent_form': 1.0}
            
            row = result.iloc[0]
            
            return {
                'attack_strength': row['total_attack'] / 2,
                'defense_strength': row['total_defense'] / 2,
                'overall_strength': row['overall_strength'],
                'recent_form': 1.0  # Would calculate from recent results
            }
            
        except Exception as e:
            logger.error(f"Error getting team performance for {team_id}: {e}")
            return {'attack_strength': 3, 'defense_strength': 3, 'recent_form': 1.0}
    
    def _calculate_transfer_confidence(self, points_improvement: float, 
                                     performance_analysis: Dict, form_analysis: Dict,
                                     availability_analysis: Dict, price_analysis: Dict) -> Dict:
        """Calculate overall confidence in the transfer recommendation."""
        try:
            confidence_score = 0.5  # Base confidence
            
            # Points improvement factor
            if points_improvement >= 3.0:
                confidence_score += 0.3
            elif points_improvement >= 1.5:
                confidence_score += 0.2
            elif points_improvement > 0:
                confidence_score += 0.1
            
            # Form factor
            if form_analysis['significant']:
                confidence_score += 0.1
            
            # Availability risk factor
            if availability_analysis['concerns']:
                confidence_score -= 0.2
            elif availability_analysis['improvements']:
                confidence_score += 0.1
            
            # Price value factor
            value_improvement = price_analysis['details'].get('value_improvement', 0)
            if value_improvement > 2.0:
                confidence_score += 0.1
            elif value_improvement < 0.5:
                confidence_score -= 0.1
            
            # Cap confidence between 0.1 and 1.0
            confidence_score = max(0.1, min(1.0, confidence_score))
            
            # Determine confidence level
            if confidence_score >= 0.8:
                level = 'Very High'
                strength = 'STRONGLY RECOMMEND'
            elif confidence_score >= 0.65:
                level = 'High'
                strength = 'RECOMMEND'
            elif confidence_score >= 0.5:
                level = 'Medium'
                strength = 'CONSIDER'
            elif confidence_score >= 0.35:
                level = 'Low'
                strength = 'WEAK OPTION'
            else:
                level = 'Very Low'
                strength = 'AVOID'
            
            return {
                'score': confidence_score,
                'level': level,
                'strength': strength
            }
            
        except Exception as e:
            logger.error(f"Error calculating transfer confidence: {e}")
            return {'score': 0.5, 'level': 'Medium', 'strength': 'CONSIDER'}
    
    def _calculate_expected_impact(self, player_out: Dict, player_in: Dict, 
                                 points_improvement: float, confidence_metrics: Dict) -> Dict:
        """Calculate expected impact of the transfer."""
        try:
            expected_points = points_improvement * confidence_metrics['score']
            
            # Risk-adjusted expectations
            conservative_estimate = expected_points * 0.8
            optimistic_estimate = expected_points * 1.2
            
            # Multi-gameweek impact (assuming transfer held for 5 gameweeks)
            gameweeks_held = 5
            total_expected_impact = expected_points * gameweeks_held
            
            impact = {
                'single_gameweek': {
                    'expected_points': expected_points,
                    'conservative': conservative_estimate,
                    'optimistic': optimistic_estimate
                },
                'multi_gameweek': {
                    'gameweeks_projected': gameweeks_held,
                    'total_expected_points': total_expected_impact,
                    'confidence_adjusted': total_expected_impact * confidence_metrics['score']
                },
                'breakeven_analysis': {
                    'gameweeks_to_payoff': max(1, abs(points_improvement)) if points_improvement != 0 else float('inf')
                }
            }
            
            return impact
            
        except Exception as e:
            logger.error(f"Error calculating expected impact: {e}")
            return {}
    
    def _generate_summary_explanation(self, explanation: Dict, points_improvement: float) -> str:
        """Generate a concise summary explanation."""
        try:
            summary_parts = []
            
            # Add primary reason
            if explanation['primary_reasons']:
                summary_parts.append(explanation['primary_reasons'][0])
            
            # Add top supporting factor
            if explanation['supporting_factors']:
                summary_parts.append(explanation['supporting_factors'][0])
            
            # Add main risk if significant
            if explanation['risk_factors']:
                summary_parts.append(f"Risk: {explanation['risk_factors'][0]}")
            
            # Add confidence assessment
            summary_parts.append(f"Confidence: {explanation['confidence_level']}")
            
            return " | ".join(summary_parts)
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return "Transfer analysis available"
    
    def _create_error_explanation(self, player_out: Dict, player_in: Dict) -> Dict:
        """Create a basic explanation when analysis fails."""
        return {
            'transfer_summary': f"{player_out['name']} → {player_in['name']}",
            'primary_reasons': ['Analysis error - manual review recommended'],
            'supporting_factors': [],
            'risk_factors': [],
            'confidence_level': 'Low',
            'recommendation_strength': 'MANUAL REVIEW',
            'detailed_analysis': {},
            'expected_impact': {},
            'summary_explanation': 'Transfer analysis failed - manual review required'
        }
    
    def generate_multi_transfer_explanation(self, transfer_combination: List[Tuple[Dict, Dict]], 
                                          predictions: Dict[int, Dict], hit_cost: int = 0) -> Dict:
        """Generate explanation for multiple transfers in combination."""
        try:
            logger.info(f"Generating multi-transfer explanation for {len(transfer_combination)} transfers")
            
            explanation = {
                'combination_summary': self._create_combination_summary(transfer_combination),
                'individual_transfers': [],
                'synergy_analysis': {},
                'overall_impact': {},
                'hit_analysis': {},
                'recommendation': {}
            }
            
            total_points_improvement = 0
            combined_reasoning = {
                'performance_upgrades': [],
                'strategic_moves': [],
                'value_improvements': [],
                'risk_factors': []
            }
            
            # Analyze each individual transfer
            for player_out, player_in in transfer_combination:
                individual_explanation = self.generate_transfer_explanation(
                    player_out, player_in, predictions
                )
                explanation['individual_transfers'].append(individual_explanation)
                
                # Accumulate overall metrics
                points_improvement = (predictions.get(player_in['id'], {}).get('points', 0) - 
                                    predictions.get(player_out['id'], {}).get('points', 0))
                total_points_improvement += points_improvement
                
                # Categorize reasons
                if points_improvement >= 2.0:
                    combined_reasoning['performance_upgrades'].append(
                        f"{player_out['name']} → {player_in['name']} (+{points_improvement:.1f})"
                    )
                elif points_improvement > 0:
                    combined_reasoning['strategic_moves'].append(
                        f"{player_out['name']} → {player_in['name']} (+{points_improvement:.1f})"
                    )
                
                # Add risk factors
                if individual_explanation['risk_factors']:
                    combined_reasoning['risk_factors'].extend(individual_explanation['risk_factors'])
            
            # Analyze transfer synergies
            explanation['synergy_analysis'] = self._analyze_transfer_synergies(
                transfer_combination, predictions
            )
            
            # Calculate overall impact
            net_improvement = total_points_improvement - hit_cost
            explanation['overall_impact'] = {
                'total_points_improvement': total_points_improvement,
                'hit_cost': hit_cost,
                'net_improvement': net_improvement,
                'combined_reasoning': combined_reasoning
            }
            
            # Hit analysis
            if hit_cost > 0:
                explanation['hit_analysis'] = self._analyze_multi_transfer_hit(
                    total_points_improvement, hit_cost, len(transfer_combination)
                )
            
            # Overall recommendation
            explanation['recommendation'] = self._generate_multi_transfer_recommendation(
                net_improvement, hit_cost, combined_reasoning
            )
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error generating multi-transfer explanation: {e}")
            return {}
    
    def _create_combination_summary(self, transfer_combination: List[Tuple[Dict, Dict]]) -> str:
        """Create summary string for transfer combination."""
        transfers = []
        for player_out, player_in in transfer_combination:
            transfers.append(f"{player_out['name']} → {player_in['name']}")
        
        return " | ".join(transfers)
    
    def _analyze_transfer_synergies(self, transfer_combination: List[Tuple[Dict, Dict]], 
                                  predictions: Dict[int, Dict]) -> Dict:
        """Analyze potential synergies between multiple transfers."""
        try:
            synergies = {
                'team_balance': [],
                'positional_upgrades': [],
                'strategic_benefits': []
            }
            
            # Check for positional balance improvements
            positions_changed = {}
            for player_out, player_in in transfer_combination:
                position = player_out.get('position', 0)
                if position not in positions_changed:
                    positions_changed[position] = []
                positions_changed[position].append((player_out, player_in))
            
            # Analyze each position's changes
            for position, changes in positions_changed.items():
                if len(changes) > 1:
                    position_name = self._get_position_name(position)
                    synergies['positional_upgrades'].append(
                        f"Complete {position_name} overhaul with {len(changes)} changes"
                    )
            
            # Check for team diversification
            teams_out = set([p_out['team'] for p_out, p_in in transfer_combination])
            teams_in = set([p_in['team'] for p_out, p_in in transfer_combination])
            
            if len(teams_in) > len(teams_out):
                synergies['team_balance'].append(
                    f"Improves team diversification ({len(teams_out)} → {len(teams_in)} teams)"
                )
            
            return synergies
            
        except Exception as e:
            logger.error(f"Error analyzing transfer synergies: {e}")
            return {}
    
    def _get_position_name(self, position_id: int) -> str:
        """Get position name from ID."""
        position_names = {1: 'Goalkeeper', 2: 'Defence', 3: 'Midfield', 4: 'Forward'}
        return position_names.get(position_id, 'Unknown')
    
    def _analyze_multi_transfer_hit(self, points_improvement: float, hit_cost: int, 
                                  num_transfers: int) -> Dict:
        """Analyze whether multi-transfer hit is justified."""
        try:
            analysis = {
                'hit_cost': hit_cost,
                'num_hits': hit_cost // 4,
                'points_improvement': points_improvement,
                'net_benefit': points_improvement - hit_cost,
                'payback_gameweeks': hit_cost / max(points_improvement, 0.1),
                'justification': ''
            }
            
            if analysis['net_benefit'] > 2:
                analysis['justification'] = f"Strongly justified: {analysis['net_benefit']:.1f} net points immediately"
            elif analysis['net_benefit'] > 0:
                analysis['justification'] = f"Marginally justified: {analysis['net_benefit']:.1f} net points"
            else:
                analysis['justification'] = f"Not justified: {analysis['net_benefit']:.1f} net points (loss)"
            
            # Add payback analysis
            if analysis['payback_gameweeks'] <= 2:
                analysis['payback_assessment'] = f"Quick payback in {analysis['payback_gameweeks']:.1f} gameweeks"
            elif analysis['payback_gameweeks'] <= 4:
                analysis['payback_assessment'] = f"Reasonable payback in {analysis['payback_gameweeks']:.1f} gameweeks"
            else:
                analysis['payback_assessment'] = f"Slow payback over {analysis['payback_gameweeks']:.1f} gameweeks"
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing multi-transfer hit: {e}")
            return {}
    
    def _generate_multi_transfer_recommendation(self, net_improvement: float, hit_cost: int,
                                              combined_reasoning: Dict) -> Dict:
        """Generate overall recommendation for multi-transfer combination."""
        try:
            if net_improvement > 3:
                strength = 'STRONGLY RECOMMEND'
                reasoning = 'Significant net improvement justifies multiple transfers'
            elif net_improvement > 1:
                strength = 'RECOMMEND'
                reasoning = 'Good net improvement with acceptable risk'
            elif net_improvement > 0:
                strength = 'CONSIDER'
                reasoning = 'Marginal improvement - consider timing and alternatives'
            else:
                strength = 'AVOID'
                reasoning = 'Negative expected value - avoid this combination'
            
            # Add hit-specific guidance
            if hit_cost > 0:
                if net_improvement > hit_cost * 0.5:
                    hit_guidance = 'Hit is justified by improvement'
                else:
                    hit_guidance = 'Hit may not be worth the cost'
            else:
                hit_guidance = 'Uses free transfers efficiently'
            
            return {
                'strength': strength,
                'reasoning': reasoning,
                'hit_guidance': hit_guidance,
                'net_improvement': net_improvement,
                'summary': f"{strength}: {reasoning}"
            }
            
        except Exception as e:
            logger.error(f"Error generating multi-transfer recommendation: {e}")
            return {}

