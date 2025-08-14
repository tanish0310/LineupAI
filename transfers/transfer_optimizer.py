import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from itertools import combinations
from sqlalchemy import create_engine
import logging
import os
from dotenv import load_dotenv
from datetime import datetime

from models.optimization.squad_optimizer import SquadOptimizer

load_dotenv()

logger = logging.getLogger(__name__)

class TransferOptimizer:
    """
    Advanced transfer optimization system that analyzes current squads and 
    recommends optimal transfers considering free transfers, hits, and future planning.
    """
    
    def __init__(self):
        self.engine = create_engine(os.getenv('DATABASE_URL'))
        self.squad_optimizer = SquadOptimizer()
        self.HIT_THRESHOLD = 4.0  # Points needed to justify -4 hit
        self.MAX_TRANSFERS_TO_ANALYZE = 3  # Max transfers in one gameweek
    
    def analyze_current_squad(self, user_squad: List[Dict], predictions: Dict[int, Dict]) -> Dict:
        """
        Comprehensive analysis of user's current squad vs optimal possibilities.
        
        Args:
            user_squad: Current squad with player data
            predictions: Player predictions for upcoming gameweek
            
        Returns:
            Detailed squad analysis with improvement potential
        """
        try:
            logger.info("Analyzing current squad performance...")
            
            # Calculate current squad predicted points
            current_predicted = sum([
                predictions.get(player['id'], {}).get('points', 0) 
                for player in user_squad
            ])
            
            # Get current squad cost and value
            current_cost = sum([player.get('current_price', player.get('cost', 0)) for player in user_squad])
            selling_value = sum([player.get('selling_price', player.get('cost', 0)) for player in user_squad])
            
            # Build optimal squad for comparison
            optimal_squad = self.squad_optimizer.build_optimal_squad(predictions)
            optimal_predicted = optimal_squad['total_predicted_points']
            
            # Calculate squad percentile ranking
            squad_percentile = self._calculate_squad_percentile(user_squad, predictions)
            
            # Identify problem areas
            problem_areas = self._identify_problem_areas(user_squad, predictions)
            
            # Calculate individual player performances vs alternatives
            player_comparisons = self._compare_players_to_alternatives(user_squad, predictions)
            
            analysis = {
                'current_performance': {
                    'predicted_points': current_predicted,
                    'squad_value': current_cost / 10,
                    'selling_value': selling_value / 10,
                    'avg_points_per_player': current_predicted / len(user_squad)
                },
                'optimal_comparison': {
                    'optimal_points': optimal_predicted,
                    'improvement_potential': optimal_predicted - current_predicted,
                    'efficiency_score': (current_predicted / optimal_predicted) * 100
                },
                'squad_ranking': {
                    'percentile': squad_percentile,
                    'rank_description': self._get_rank_description(squad_percentile)
                },
                'problem_areas': problem_areas,
                'player_comparisons': player_comparisons,
                'overall_assessment': self._generate_overall_assessment(
                    current_predicted, optimal_predicted, squad_percentile, problem_areas
                )
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing current squad: {e}")
            raise
    
    def suggest_transfers(self, current_squad: List[Dict], free_transfers: int, 
                         predictions: Dict[int, Dict], budget_remaining: float = 0) -> List[Dict]:
        """
        Generate optimal transfer recommendations considering free transfers and hits.
        
        Args:
            current_squad: Current squad with selling prices
            free_transfers: Number of free transfers available
            predictions: Player predictions
            budget_remaining: Available budget for transfers
            
        Returns:
            List of transfer recommendations ranked by net improvement
        """
        try:
            logger.info(f"Generating transfer suggestions with {free_transfers} free transfers...")
            
            transfer_options = []
            
            # Convert budget to tenths for internal calculations
            budget_tenths = int(budget_remaining * 10)
            
            # Analyze different transfer scenarios (1, 2, 3 transfers)
            for num_transfers in range(1, min(self.MAX_TRANSFERS_TO_ANALYZE + 1, len(current_squad) + 1)):
                logger.info(f"Analyzing {num_transfers}-transfer options...")
                
                # Get all combinations of players to transfer out
                for players_out in combinations(current_squad, num_transfers):
                    # Calculate available budget after sales
                    budget_from_sales = sum([p.get('selling_price', p.get('cost', 0)) for p in players_out])
                    total_budget = budget_tenths + budget_from_sales
                    
                    # Find optimal replacements
                    replacement_options = self._find_optimal_replacements(
                        players_out, current_squad, total_budget, predictions
                    )
                    
                    for players_in in replacement_options:
                        # Validate the transfer
                        if self._is_valid_transfer(players_out, players_in, current_squad, total_budget):
                            # Calculate transfer metrics
                            transfer_metrics = self._calculate_transfer_metrics(
                                players_out, players_in, predictions, num_transfers, free_transfers
                            )
                            
                            transfer_options.append({
                                'players_out': players_out,
                                'players_in': players_in,
                                'num_transfers': num_transfers,
                                'points_improvement': transfer_metrics['points_improvement'],
                                'cost_change': transfer_metrics['cost_change'],
                                'hit_cost': transfer_metrics['hit_cost'],
                                'net_improvement': transfer_metrics['net_improvement'],
                                'reasoning': self._generate_transfer_reasoning(players_out, players_in, predictions),
                                'confidence': transfer_metrics['confidence'],
                                'priority': transfer_metrics['priority']
                            })
            
            # Sort by net improvement and return top options
            transfer_options.sort(key=lambda x: x['net_improvement'], reverse=True)
            
            # Filter and rank recommendations
            recommendations = self._filter_and_rank_transfers(transfer_options, free_transfers)
            
            logger.info(f"Generated {len(recommendations)} transfer recommendations")
            return recommendations
            
        except Exception as e:
            logger.error(f"Error suggesting transfers: {e}")
            raise
    
    def _find_optimal_replacements(self, players_out: Tuple, current_squad: List[Dict],
                                 budget: int, predictions: Dict[int, Dict]) -> List[Tuple]:
        """Find optimal replacement combinations for players being transferred out."""
        try:
            num_replacements = len(players_out)
            
            # Get position requirements for replacements
            positions_needed = [p['position'] for p in players_out]
            teams_to_avoid = set()
            
            # Calculate team constraints
            remaining_squad = [p for p in current_squad if p not in players_out]
            team_counts = {}
            for player in remaining_squad:
                team = player['team']
                team_counts[team] = team_counts.get(team, 0) + 1
            
            # Get available players for each position
            available_players = self._get_available_replacement_players(
                positions_needed, team_counts, predictions
            )
            
            # Generate valid replacement combinations
            replacement_combinations = []
            
            if num_replacements == 1:
                position = positions_needed[0]
                for player in available_players.get(position, []):
                    if player['cost'] <= budget:
                        replacement_combinations.append((player,))
            
            elif num_replacements == 2:
                pos1, pos2 = positions_needed
                for p1 in available_players.get(pos1, []):
                    for p2 in available_players.get(pos2, []):
                        if (p1['cost'] + p2['cost'] <= budget and 
                            p1['id'] != p2['id'] and
                            self._check_team_constraints(remaining_squad + [p1, p2])):
                            replacement_combinations.append((p1, p2))
            
            elif num_replacements == 3:
                pos1, pos2, pos3 = positions_needed
                for p1 in available_players.get(pos1, []):
                    for p2 in available_players.get(pos2, []):
                        for p3 in available_players.get(pos3, []):
                            if (p1['cost'] + p2['cost'] + p3['cost'] <= budget and
                                len(set([p1['id'], p2['id'], p3['id']])) == 3 and
                                self._check_team_constraints(remaining_squad + [p1, p2, p3])):
                                replacement_combinations.append((p1, p2, p3))
            
            # Sort by total predicted points and return top options
            replacement_combinations.sort(
                key=lambda combo: sum([p['predicted_points'] for p in combo]),
                reverse=True
            )
            
            return replacement_combinations[:20]  # Limit to top 20 combinations
            
        except Exception as e:
            logger.error(f"Error finding optimal replacements: {e}")
            return []
    
    def _get_available_replacement_players(self, positions_needed: List[int], team_counts: Dict, predictions: Dict[int, Dict]) -> Dict[int, List[Dict]]:
        """Get available players for replacement by position."""
        try:
            available_players = {}
        
            for position in set(positions_needed):
                # Simple query - get all players for this position
                query = """
                SELECT 
                    p.id, p.web_name, p.position, p.team, p.now_cost,
                    p.total_points, p.form, t.name as team_name
                FROM players p
                JOIN teams t ON p.team = t.id
                WHERE p.position = %s
                AND p.status = 'a'
                ORDER BY p.total_points DESC
                LIMIT 50
                """
            
                players_df = pd.read_sql(query, self.engine, params=[position])
            
                # Filter for players with predictions (simple check)
                players_list = []
                for _, player in players_df.iterrows():
                    if player['id'] in predictions:  # Only include players with predictions
                        player_dict = {
                            'id': player['id'],
                            'name': player['web_name'],
                            'position': player['position'],
                            'team': player['team'],
                            'team_name': player['team_name'],
                            'cost': player['now_cost'],
                            'predicted_points': predictions.get(player['id'], {}).get('points', 0),
                            'current_points': player['total_points'],
                            'form': player['form']
                        }
                    
                        # Check team constraints
                        current_team_count = team_counts.get(player['team'], 0)
                        if current_team_count < 3:  # Can still add players from this team
                            players_list.append(player_dict)
            
                available_players[position] = players_list
        
            return available_players
        
        except Exception as e:
            logger.error(f"Error getting available replacement players: {e}")
            return {}

    
    def _check_team_constraints(self, squad: List[Dict]) -> bool:
        """Check if squad satisfies team constraints (max 3 per team)."""
        team_counts = {}
        for player in squad:
            team = player['team']
            team_counts[team] = team_counts.get(team, 0) + 1
            if team_counts[team] > 3:
                return False
        return True
    
    def _is_valid_transfer(self, players_out: Tuple, players_in: Tuple, 
                         current_squad: List[Dict], budget: int) -> bool:
        """Validate if a transfer combination is legal."""
        try:
            # Check budget
            cost_out = sum([p.get('selling_price', p.get('cost', 0)) for p in players_out])
            cost_in = sum([p['cost'] for p in players_in])
            
            if cost_in > cost_out + budget:
                return False
            
            # Check positions match
            positions_out = sorted([p['position'] for p in players_out])
            positions_in = sorted([p['position'] for p in players_in])
            
            if positions_out != positions_in:
                return False
            
            # Check resulting squad is valid
            remaining_squad = [p for p in current_squad if p not in players_out]
            new_squad = remaining_squad + list(players_in)
            
            # Validate new squad
            validation = self.squad_optimizer.validate_squad(new_squad)
            
            return validation['is_valid']
            
        except Exception as e:
            logger.error(f"Error validating transfer: {e}")
            return False
    
    def _calculate_transfer_metrics(self, players_out: Tuple, players_in: Tuple,
                                  predictions: Dict[int, Dict], num_transfers: int,
                                  free_transfers: int) -> Dict:
        """Calculate comprehensive metrics for a transfer combination."""
        try:
            # Points improvement
            points_out = sum([predictions.get(p['id'], {}).get('points', 0) for p in players_out])
            points_in = sum([p['predicted_points'] for p in players_in])
            points_improvement = points_in - points_out
            
            # Cost change
            cost_out = sum([p.get('selling_price', p.get('cost', 0)) for p in players_out])
            cost_in = sum([p['cost'] for p in players_in])
            cost_change = (cost_in - cost_out) / 10  # Convert to millions
            
            # Hit calculation
            hits_required = max(0, num_transfers - free_transfers)
            hit_cost = hits_required * 4
            
            # Net improvement
            net_improvement = points_improvement - hit_cost
            
            # Confidence calculation
            confidence_out = sum([predictions.get(p['id'], {}).get('confidence', 0.5) for p in players_out]) / len(players_out)
            confidence_in = sum([predictions.get(p['id'], {}).get('confidence', 0.5) for p in players_in]) / len(players_in)
            confidence = (confidence_in + (1 - confidence_out)) / 2
            
            # Priority calculation
            if net_improvement > 2:
                priority = 'High'
            elif net_improvement > 0:
                priority = 'Medium'
            else:
                priority = 'Low'
            
            return {
                'points_improvement': points_improvement,
                'cost_change': cost_change,
                'hit_cost': hit_cost,
                'net_improvement': net_improvement,
                'confidence': confidence,
                'priority': priority
            }
            
        except Exception as e:
            logger.error(f"Error calculating transfer metrics: {e}")
            return {}
    
    def _generate_transfer_reasoning(self, players_out: Tuple, players_in: Tuple,
                                   predictions: Dict[int, Dict]) -> List[str]:
        """Generate human-readable reasoning for transfer recommendations."""
        try:
            reasoning = []
            
            for player_out, player_in in zip(players_out, players_in):
                points_diff = player_in['predicted_points'] - predictions.get(player_out['id'], {}).get('points', 0)
                
                reason_parts = []
                
                # Points improvement
                if points_diff > 2:
                    reason_parts.append(f"+{points_diff:.1f} points improvement")
                elif points_diff > 0:
                    reason_parts.append(f"+{points_diff:.1f} points gain")
                else:
                    reason_parts.append("strategic move")
                
                # Form comparison
                form_in = player_in.get('form', 0)
                form_out = player_out.get('form', 0)
                
                if form_in > form_out + 1:
                    reason_parts.append("superior form")
                
                # Cost efficiency
                cost_diff = player_in['cost'] - player_out.get('cost', 0)
                if cost_diff < 0:
                    reason_parts.append(f"frees up £{abs(cost_diff)/10:.1f}m")
                
                transfer_reason = f"{player_out['name']} → {player_in['name']}: {', '.join(reason_parts)}"
                reasoning.append(transfer_reason)
            
            return reasoning
            
        except Exception as e:
            logger.error(f"Error generating transfer reasoning: {e}")
            return ["Transfer reasoning unavailable"]
    
    def _filter_and_rank_transfers(self, transfer_options: List[Dict], free_transfers: int) -> List[Dict]:
        """Filter and rank transfer recommendations by value and feasibility."""
        try:
            recommendations = []
            
            for option in transfer_options:
                # Only recommend transfers with positive net improvement
                if option['net_improvement'] > 0:
                    # Add recommendation strength
                    if option['net_improvement'] > 3:
                        option['recommendation'] = 'STRONGLY RECOMMENDED'
                    elif option['net_improvement'] > 1:
                        option['recommendation'] = 'RECOMMENDED'
                    else:
                        option['recommendation'] = 'CONSIDER'
                    
                    # Add hit analysis
                    if option['hit_cost'] > 0:
                        option['hit_analysis'] = f"Requires {option['hit_cost']//4} transfer hit(s) (-{option['hit_cost']} points)"
                        
                        if option['net_improvement'] > 1:
                            option['hit_recommendation'] = 'TAKE HIT'
                        else:
                            option['hit_recommendation'] = 'MARGINAL - CONSIDER WAITING'
                    else:
                        option['hit_analysis'] = 'Uses free transfer(s)'
                        option['hit_recommendation'] = 'MAKE TRANSFER'
                    
                    recommendations.append(option)
            
            # Sort by net improvement
            recommendations.sort(key=lambda x: x['net_improvement'], reverse=True)
            
            return recommendations[:10]  # Return top 10 recommendations
            
        except Exception as e:
            logger.error(f"Error filtering and ranking transfers: {e}")
            return []
    
    def evaluate_transfer_hits(self, transfer_options: List[Dict]) -> List[Dict]:
        """Evaluate whether transfer hits are worth taking."""
        try:
            hit_evaluations = []
            
            for option in transfer_options:
                if option['hit_cost'] > 0:
                    evaluation = {
                        'transfer': option,
                        'hit_cost': option['hit_cost'],
                        'points_improvement': option['points_improvement'],
                        'net_benefit': option['net_improvement'],
                        'payback_weeks': option['hit_cost'] / max(option['points_improvement'], 0.1),
                        'recommendation': 'TAKE' if option['net_improvement'] > 1 else 'AVOID',
                        'reasoning': self._generate_hit_reasoning(option)
                    }
                    
                    hit_evaluations.append(evaluation)
            
            return hit_evaluations
            
        except Exception as e:
            logger.error(f"Error evaluating transfer hits: {e}")
            return []
    
    def _generate_hit_reasoning(self, option: Dict) -> str:
        """Generate reasoning for hit recommendations."""
        net_benefit = option['net_improvement']
        hit_cost = option['hit_cost']
        
        if net_benefit > 2:
            return f"Strong benefit (+{net_benefit:.1f} net points) justifies {hit_cost//4} hit(s)"
        elif net_benefit > 0:
            return f"Marginal benefit (+{net_benefit:.1f} net points) may justify hit"
        else:
            return f"Negative return ({net_benefit:.1f} net points) - avoid hit"
    
    def multi_gameweek_transfer_planning(self, current_squad: List[Dict], 
                                       predictions_3gw: Dict[int, Dict]) -> Dict:
        """Plan optimal transfers over next 3 gameweeks."""
        try:
            logger.info("Planning multi-gameweek transfer strategy...")
            
            # This would analyze fixture swings, player form trends, and price changes
            # For now, providing framework
            
            plan = {
                'current_gameweek': {
                    'priority_transfers': [],
                    'reasoning': "Focus on immediate points"
                },
                'next_gameweek': {
                    'planned_transfers': [],
                    'reasoning': "Consider fixture difficulty changes"
                },
                'gameweek_plus_2': {
                    'strategic_moves': [],
                    'reasoning': "Position for fixture swings"
                },
                'overall_strategy': "Maximize points while preparing for fixture changes"
            }
            
            return plan
            
        except Exception as e:
            logger.error(f"Error in multi-gameweek planning: {e}")
            return {}
    
    # Helper methods for analysis
    
    def _calculate_squad_percentile(self, squad: List[Dict], predictions: Dict[int, Dict]) -> float:
        """Calculate what percentile the squad ranks in terms of predicted points."""
        try:
            squad_points = sum([predictions.get(p['id'], {}).get('points', 0) for p in squad])
            
            # This would compare against all possible squads or a sample
            # For now, using a rough approximation
            
            if squad_points >= 65:
                return 95  # Top 5%
            elif squad_points >= 60:
                return 85  # Top 15%
            elif squad_points >= 55:
                return 70  # Top 30%
            elif squad_points >= 50:
                return 50  # Average
            else:
                return 25  # Below average
                
        except Exception:
            return 50
    
    def _get_rank_description(self, percentile: float) -> str:
        """Get description of squad ranking."""
        if percentile >= 90:
            return "Excellent - Top 10%"
        elif percentile >= 75:
            return "Good - Top 25%"
        elif percentile >= 50:
            return "Average - Top 50%"
        elif percentile >= 25:
            return "Below Average"
        else:
            return "Poor - Bottom 25%"
    
    def _identify_problem_areas(self, squad: List[Dict], predictions: Dict[int, Dict]) -> List[Dict]:
        """Identify problematic players in the squad."""
        problems = []
        
        for player in squad:
            predicted_points = predictions.get(player['id'], {}).get('points', 0)
            
            if predicted_points < 3:
                problems.append({
                    'player': player['name'],
                    'issue': 'Low predicted points',
                    'severity': 'High',
                    'predicted_points': predicted_points
                })
            elif predicted_points < 4:
                problems.append({
                    'player': player['name'],
                    'issue': 'Below average performance',
                    'severity': 'Medium',
                    'predicted_points': predicted_points
                })
        
        return problems
    
    def _compare_players_to_alternatives(self, squad: List[Dict], predictions: Dict[int, Dict]) -> List[Dict]:
        """Compare each squad player to best alternatives in their position."""
        comparisons = []
        
        # This would compare each player to the top options in their position
        # Placeholder implementation
        
        for player in squad:
            comparison = {
                'player': player['name'],
                'position': player.get('position_name', 'Unknown'),
                'predicted_points': predictions.get(player['id'], {}).get('points', 0),
                'rank_in_position': 'TBD',  # Would calculate actual rank
                'best_alternative': 'TBD',  # Would find best alternative
                'points_difference': 0      # Difference to best alternative
            }
            comparisons.append(comparison)
        
        return comparisons
    
    def _generate_overall_assessment(self, current_points: float, optimal_points: float,
                                   percentile: float, problems: List[Dict]) -> Dict:
        """Generate overall squad assessment."""
        gap = optimal_points - current_points
        
        if gap < 3:
            assessment = "Excellent squad with minimal improvement needed"
            priority = "Low"
        elif gap < 6:
            assessment = "Good squad with some room for improvement"
            priority = "Medium"
        else:
            assessment = "Squad has significant improvement potential"
            priority = "High"
        
        return {
            'assessment': assessment,
            'improvement_priority': priority,
            'points_gap': gap,
            'major_issues': len([p for p in problems if p['severity'] == 'High'])
        }
    
    
