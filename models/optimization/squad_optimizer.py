import pandas as pd
import numpy as np
from pulp import *
import logging
from typing import Dict, List, Tuple, Optional
from sqlalchemy import create_engine
from datetime import datetime
import os
from dotenv import load_dotenv
from itertools import combinations

load_dotenv()

logger = logging.getLogger(__name__)

class SquadOptimizer:
    """
    Advanced squad optimization system using linear programming to build
    optimal 15-man squads and starting XIs within FPL constraints.
    """
    
    def __init__(self):
        self.engine = create_engine(os.getenv('DATABASE_URL'))
        
        # FPL constraints
        self.BUDGET = 1000  # £100.0m in tenths
        self.SQUAD_SIZE = 15
        self.STARTING_XI_SIZE = 11
        
        # Position requirements
        self.POSITIONS = {
            1: {'name': 'GK', 'min': 2, 'max': 2, 'starting_min': 1, 'starting_max': 1},
            2: {'name': 'DEF', 'min': 5, 'max': 5, 'starting_min': 3, 'starting_max': 5},
            3: {'name': 'MID', 'min': 5, 'max': 5, 'starting_min': 2, 'starting_max': 5},
            4: {'name': 'FWD', 'min': 3, 'max': 3, 'starting_min': 1, 'starting_max': 3}
        }
        
        self.MAX_PER_TEAM = 3
        
        # Valid formations (GK, DEF, MID, FWD)
        self.VALID_FORMATIONS = [
            (1, 3, 4, 3), (1, 3, 5, 2), (1, 4, 3, 3),
            (1, 4, 4, 2), (1, 4, 5, 1), (1, 5, 3, 2),
            (1, 5, 4, 1)
        ]
    
    def build_optimal_squad(self, predictions: Dict[int, Dict], budget: int = 1000, 
                          existing_squad: List[int] = None, locked_players: List[int] = None) -> Dict:
        """
        Build optimal 15-man squad using linear programming.
        
        Args:
            predictions: Player predictions {player_id: {'points': float, 'confidence': float}}
            budget: Available budget in tenths (default 100.0m)
            existing_squad: List of current player IDs (for transfers)
            locked_players: Players that must be included
            
        Returns:
            Complete squad solution with starting XI and bench
        """
        try:
            logger.info("Building optimal squad using linear programming...")
            
            # Get available players with their data
            players_df = self._get_available_players(predictions)
            
            if players_df.empty:
                raise ValueError("No available players found")
            
            # Create optimization problem
            prob = LpProblem("FPL_Squad_Optimization", LpMaximize)
            
            # Decision variables: binary variable for each player
            player_vars = {}
            for _, player in players_df.iterrows():
                player_vars[player['id']] = LpVariable(
                    f"player_{player['id']}", 
                    cat='Binary'
                )
            
            # Objective function: maximize total predicted points
            prob += lpSum([
                predictions[player_id]['points'] * player_vars[player_id] 
                for player_id in player_vars
                if player_id in predictions
            ]), "Total_Predicted_Points"
            
            # Add constraints
            self._add_squad_constraints(prob, player_vars, players_df, budget, 
                                      locked_players, existing_squad)
            
            # Solve the problem
            prob.solve(PULP_CBC_CMD(msg=0))
            
            # Extract solution
            if prob.status == LpStatusOptimal:
                solution = self._extract_squad_solution(prob, player_vars, players_df, predictions)
                
                # Optimize starting XI from the 15-man squad
                solution['starting_xi'] = self.optimize_starting_xi(
                    solution['squad_15'], predictions
                )
                
                # Add captain recommendations
                solution['captain_options'] = self._recommend_captains(
                    solution['starting_xi'], predictions
                )
                
                logger.info(f"Optimal squad built: {solution['total_predicted_points']:.1f} points")
                return solution
            else:
                raise ValueError(f"Optimization failed with status: {LpStatus[prob.status]}")
                
        except Exception as e:
            logger.error(f"Error building optimal squad: {e}")
            raise
    
    def _get_available_players(self, predictions: Dict[int, Dict]) -> pd.DataFrame:
        """Get all available players with their current data."""
        try:
            # Get players who have predictions and are available
            player_ids = list(predictions.keys())
        
            if not player_ids:
                return pd.DataFrame()
        
            # Convert to format suitable for SQL IN clause
            ids_str = ','.join(map(str, player_ids))
        
            # Simplified query without positions table JOIN (which might be causing issues)
            query = f"""
            SELECT 
                p.id,
                p.web_name,
                p.position,
                p.team,
                p.now_cost,
                p.total_points,
                p.form,
                p.selected_by_percent,
                p.status,
                t.name as team_name,
                CASE 
                    WHEN p.position = 1 THEN 'GK'
                    WHEN p.position = 2 THEN 'DEF'
                    WHEN p.position = 3 THEN 'MID'
                    WHEN p.position = 4 THEN 'FWD'
                END as position_name
            FROM players p
            JOIN teams t ON p.team = t.id
            WHERE p.id IN ({ids_str})
            AND p.status = 'a'
            ORDER BY p.position, p.total_points DESC
            """
        
            players_df = pd.read_sql(query, self.engine)
        
            # Add prediction data
            players_df['predicted_points'] = players_df['id'].map(
                lambda x: predictions.get(x, {}).get('points', 0)
            )
            players_df['confidence'] = players_df['id'].map(
                lambda x: predictions.get(x, {}).get('confidence', 0.5)
            )
        
            logger.info(f"Retrieved {len(players_df)} players for optimization")
            return players_df
        
        except Exception as e:
            logger.error(f"Error getting available players: {e}")
            return pd.DataFrame()
        
    def debug_optimization_constraints(self, predictions: Dict[int, Dict]) -> Dict:
        """Debug optimization constraints to identify infeasibility issues."""
        try:
            players_df = self._get_available_players(predictions)
        
            if players_df.empty:
                return {'error': 'No players available'}
        
            debug_info = {
                'total_players': len(players_df),
                'players_by_position': players_df['position'].value_counts().to_dict(),
                'players_by_team': players_df['team'].value_counts().head(10).to_dict(),
                'price_stats': {
                    'min_cost': players_df['now_cost'].min() / 10,
                    'max_cost': players_df['now_cost'].max() / 10,
                    'avg_cost': players_df['now_cost'].mean() / 10,
                    'total_budget': self.BUDGET / 10
                },
                'constraint_feasibility': {}
            }
        
            # Check position feasibility
            for position_id, requirements in self.POSITIONS.items():
                available = len(players_df[players_df['position'] == position_id])
                debug_info['constraint_feasibility'][f'position_{position_id}'] = {
                    'available': available,
                    'required_min': requirements['min'],
                    'required_max': requirements['max'],
                    'feasible': available >= requirements['min']
                }
        
            # Check cheapest possible squad cost
            cheapest_squad_cost = 0
            for position_id, requirements in self.POSITIONS.items():
                position_players = players_df[players_df['position'] == position_id]
                if len(position_players) >= requirements['min']:
                    cheapest_players = position_players.nsmallest(requirements['min'], 'now_cost')
                    cheapest_squad_cost += cheapest_players['now_cost'].sum()
                else:
                    debug_info['constraint_feasibility']['budget'] = {
                        'error': f'Insufficient players for position {position_id}'
                    }
                    return debug_info
        
            debug_info['constraint_feasibility']['budget'] = {
                'cheapest_possible_squad': cheapest_squad_cost / 10,
                'available_budget': self.BUDGET / 10,
                'feasible': cheapest_squad_cost <= self.BUDGET
            }
        
            return debug_info
        
        except Exception as e:
            logger.error(f"Error debugging constraints: {e}")
            return {'error': str(e)}


    
    def _add_squad_constraints(self, prob, player_vars: Dict, players_df: pd.DataFrame, budget: int, locked_players: List[int] = None, existing_squad: List[int] = None):
        """Add all FPL constraints to the optimization problem."""
    
        # 1. Squad size constraint (exactly 15 players)
        prob += lpSum([player_vars[p_id] for p_id in player_vars]) == self.SQUAD_SIZE, "Squad_Size"
    
        # 2. Budget constraint - FIXED VERSION
        prob += lpSum([
            players_df.set_index('id').loc[p_id, 'now_cost'] * player_vars[p_id]
            for p_id in player_vars
            if p_id in players_df['id'].values  # Add this safety check
        ]) <= budget, "Budget_Constraint"
    
        # 3. Position constraints - FIXED VERSION
        for position_id, requirements in self.POSITIONS.items():
            position_players = players_df[players_df['position'] == position_id]['id'].tolist()
            position_vars = [player_vars[p_id] for p_id in position_players if p_id in player_vars]
        
            if position_vars:  # Only add constraint if we have players
                # Use exact constraints for fixed positions
                prob += lpSum(position_vars) == requirements['min'], f"Exact_{requirements['name']}"
    
        # 4. Team constraints (max 3 players per team) - RELAXED FOR TESTING
        teams = players_df['team'].unique()
        for team_id in teams:
            team_players = players_df[players_df['team'] == team_id]['id'].tolist()
            team_vars = [player_vars[p_id] for p_id in team_players if p_id in player_vars]
        
            if len(team_vars) > self.MAX_PER_TEAM:  # Only constrain teams with enough players
                prob += lpSum(team_vars) <= self.MAX_PER_TEAM, f"Team_{team_id}_Limit"
    
        # 5. Locked players constraints (must be included)
        if locked_players:
            for player_id in locked_players:
                if player_id in player_vars:
                    prob += player_vars[player_id] == 1, f"Locked_Player_{player_id}"

    
    def _extract_squad_solution(self, prob, player_vars: Dict, players_df: pd.DataFrame,
                              predictions: Dict[int, Dict]) -> Dict:
        """Extract the optimal squad solution from the solved problem."""
        try:
            selected_players = []
            total_cost = 0
            total_predicted_points = 0
            
            for player_id, var in player_vars.items():
                if var.varValue == 1:  # Player is selected
                    player_info = players_df[players_df['id'] == player_id].iloc[0]
                    prediction_info = predictions.get(player_id, {})
                    
                    player_data = {
                        'id': player_id,
                        'name': player_info['web_name'],
                        'position': player_info['position'],
                        'position_name': player_info['position_name'],
                        'team': player_info['team'],
                        'team_name': player_info['team_name'],
                        'cost': player_info['now_cost'],
                        'predicted_points': prediction_info.get('points', 0),
                        'confidence': prediction_info.get('confidence', 0.5),
                        'current_points': player_info['total_points'],
                        'form': player_info['form']
                    }
                    
                    selected_players.append(player_data)
                    total_cost += player_info['now_cost']
                    total_predicted_points += prediction_info.get('points', 0)
            
            # Organize by position
            squad_by_position = {
                1: [p for p in selected_players if p['position'] == 1],  # GK
                2: [p for p in selected_players if p['position'] == 2],  # DEF
                3: [p for p in selected_players if p['position'] == 3],  # MID
                4: [p for p in selected_players if p['position'] == 4]   # FWD
            }
            
            solution = {
                'squad_15': selected_players,
                'squad_by_position': squad_by_position,
                'total_cost': total_cost / 10,  # Convert back to millions
                'budget_remaining': (self.BUDGET - total_cost) / 10,
                'total_predicted_points': total_predicted_points,
                'squad_size': len(selected_players),
                'optimization_status': 'optimal'
            }
            
            return solution
            
        except Exception as e:
            logger.error(f"Error extracting squad solution: {e}")
            raise
    
    def optimize_starting_xi(self, squad_15: List[Dict], predictions: Dict[int, Dict]) -> Dict:
        """
        Optimize starting XI selection from 15-man squad.
        Tests all valid formations to find the highest-scoring combination.
        """
        try:
            logger.info("Optimizing starting XI from 15-man squad...")
            
            best_xi = None
            best_points = 0
            best_formation = None
            
            # Group players by position
            players_by_position = {
                1: [p for p in squad_15 if p['position'] == 1],  # GK
                2: [p for p in squad_15 if p['position'] == 2],  # DEF
                3: [p for p in squad_15 if p['position'] == 3],  # MID
                4: [p for p in squad_15 if p['position'] == 4]   # FWD
            }
            
            # Try each valid formation
            for formation in self.VALID_FORMATIONS:
                gk_count, def_count, mid_count, fwd_count = formation
                
                # Select best players for this formation
                xi_attempt = self._select_xi_for_formation(
                    players_by_position, formation, predictions
                )
                
                if xi_attempt:
                    total_points = sum([p['predicted_points'] for p in xi_attempt['players']])
                    
                    if total_points > best_points:
                        best_points = total_points
                        best_xi = xi_attempt
                        best_formation = formation
            
            if best_xi:
                # Add bench players
                xi_player_ids = [p['id'] for p in best_xi['players']]
                bench = [p for p in squad_15 if p['id'] not in xi_player_ids]
                
                # Order bench by predicted points (first sub most likely to come on)
                bench.sort(key=lambda x: x['predicted_points'], reverse=True)
                
                result = {
                    'formation': f"{best_formation[1]}-{best_formation[2]}-{best_formation[3]}",
                    'formation_tuple': best_formation,
                    'players': best_xi['players'],
                    'players_by_position': best_xi['players_by_position'],
                    'bench': bench,
                    'total_predicted_points': best_points,
                    'bench_points': sum([p['predicted_points'] for p in bench])
                }
                
                logger.info(f"Optimal starting XI: {result['formation']} formation, {best_points:.1f} points")
                return result
            else:
                raise ValueError("Could not find valid starting XI")
                
        except Exception as e:
            logger.error(f"Error optimizing starting XI: {e}")
            raise
    
    def _select_xi_for_formation(self, players_by_position: Dict, formation: Tuple,
                               predictions: Dict[int, Dict]) -> Optional[Dict]:
        """Select best XI for a specific formation."""
        try:
            gk_count, def_count, mid_count, fwd_count = formation
            required_counts = {1: gk_count, 2: def_count, 3: mid_count, 4: fwd_count}
            
            selected_players = []
            xi_by_position = {1: [], 2: [], 3: [], 4: []}
            
            # Select best players for each position based on predicted points
            for position, count_needed in required_counts.items():
                available_players = players_by_position.get(position, [])
                
                if len(available_players) < count_needed:
                    return None  # Not enough players for this formation
                
                # Sort by predicted points and select top players
                sorted_players = sorted(
                    available_players,
                    key=lambda x: x['predicted_points'],
                    reverse=True
                )
                
                position_selections = sorted_players[:count_needed]
                selected_players.extend(position_selections)
                xi_by_position[position] = position_selections
            
            return {
                'players': selected_players,
                'players_by_position': xi_by_position,
                'formation': formation
            }
            
        except Exception as e:
            logger.error(f"Error selecting XI for formation {formation}: {e}")
            return None
    
    def _recommend_captains(self, starting_xi: Dict, predictions: Dict[int, Dict]) -> List[Dict]:
        """Recommend top 3 captain options from starting XI."""
        try:
            captain_candidates = []
        
            for player in starting_xi['players']:
                player_id = player['id']
                predicted_points = player['predicted_points']
                confidence = player['confidence']
            
                # Simple captain scoring logic (avoiding circular import)
                base_score = predicted_points * 2  # Captain gets double points
            
                # Adjust for confidence and position
                position_multipliers = {1: 0.5, 2: 0.7, 3: 1.0, 4: 1.2}  # Forwards/Mids favored
                position_multiplier = position_multipliers.get(player['position'], 1.0)
            
                captain_score = base_score * confidence * position_multiplier
            
                captain_candidates.append({
                    'player_id': player_id,
                    'name': player['name'],
                    'position_name': player['position_name'],
                    'predicted_points': predicted_points,
                    'captain_score': captain_score,
                    'expected_captain_points': predicted_points * 2,
                    'safety_score': confidence,
                    'upside_potential': predicted_points * 1.5,
                    'reasoning': f"High predicted points ({predicted_points:.1f}) with {confidence:.1%} confidence"
                })
        
            # Sort by captain score and return top 3
            captain_candidates.sort(key=lambda x: x['captain_score'], reverse=True)
        
            return captain_candidates[:3]
        
        except Exception as e:
            logger.error(f"Error recommending captains: {e}")
            return []

    
    def generate_squad_analysis(self, squad_solution: Dict) -> Dict:
        """Generate comprehensive analysis of the squad solution."""
        try:
            analysis = {
                'squad_summary': {
                    'total_players': len(squad_solution['squad_15']),
                    'total_cost': squad_solution['total_cost'],
                    'budget_remaining': squad_solution['budget_remaining'],
                    'predicted_points': squad_solution['total_predicted_points'],
                    'avg_points_per_player': squad_solution['total_predicted_points'] / 15
                },
                'position_breakdown': {},
                'team_distribution': {},
                'price_analysis': {},
                'form_analysis': {},
                'risk_assessment': {}
            }
            
            # Position breakdown
            for position_id, players in squad_solution['squad_by_position'].items():
                position_name = self.POSITIONS[position_id]['name']
                analysis['position_breakdown'][position_name] = {
                    'count': len(players),
                    'total_cost': sum([p['cost'] for p in players]) / 10,
                    'total_predicted_points': sum([p['predicted_points'] for p in players]),
                    'avg_predicted_points': sum([p['predicted_points'] for p in players]) / len(players) if players else 0,
                    'players': [{'name': p['name'], 'cost': p['cost']/10, 'points': p['predicted_points']} for p in players]
                }
            
            # Team distribution
            team_distribution = {}
            for player in squad_solution['squad_15']:
                team = player['team_name']
                if team not in team_distribution:
                    team_distribution[team] = []
                team_distribution[team].append({
                    'name': player['name'],
                    'position': player['position_name'],
                    'predicted_points': player['predicted_points']
                })
            
            analysis['team_distribution'] = team_distribution
            
            # Price analysis
            costs = [p['cost']/10 for p in squad_solution['squad_15']]
            analysis['price_analysis'] = {
                'most_expensive': max(costs),
                'cheapest': min(costs),
                'average_cost': sum(costs) / len(costs),
                'premium_players': len([c for c in costs if c >= 9.0]),  # £9.0m+
                'budget_players': len([c for c in costs if c <= 5.0])   # £5.0m or less
            }
            
            # Form analysis
            forms = [p['form'] for p in squad_solution['squad_15']]
            analysis['form_analysis'] = {
                'avg_form': sum(forms) / len(forms),
                'excellent_form': len([f for f in forms if f >= 4.0]),  # Form 4.0+
                'poor_form': len([f for f in forms if f < 2.0])         # Form < 2.0
            }
            
            # Risk assessment
            confidences = [p['confidence'] for p in squad_solution['squad_15']]
            analysis['risk_assessment'] = {
                'avg_confidence': sum(confidences) / len(confidences),
                'high_confidence': len([c for c in confidences if c >= 0.8]),
                'risky_picks': len([c for c in confidences if c < 0.5]),
                'overall_risk_level': 'Low' if sum(confidences) / len(confidences) > 0.7 else 'Medium' if sum(confidences) / len(confidences) > 0.5 else 'High'
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error generating squad analysis: {e}")
            return {}
    
    def validate_squad(self, squad: List[Dict]) -> Dict:
        """Validate that a squad meets all FPL constraints."""
        try:
            validation = {
                'is_valid': True,
                'violations': [],
                'summary': {}
            }
            
            # Check squad size
            if len(squad) != 15:
                validation['is_valid'] = False
                validation['violations'].append(f"Squad size is {len(squad)}, must be 15")
            
            # Check position requirements
            position_counts = {1: 0, 2: 0, 3: 0, 4: 0}
            for player in squad:
                position_counts[player['position']] += 1
            
            for position_id, requirements in self.POSITIONS.items():
                count = position_counts[position_id]
                if count < requirements['min'] or count > requirements['max']:
                    validation['is_valid'] = False
                    validation['violations'].append(
                        f"{requirements['name']}: {count} players (required: {requirements['min']}-{requirements['max']})"
                    )
            
            # Check team limits
            team_counts = {}
            for player in squad:
                team = player['team']
                team_counts[team] = team_counts.get(team, 0) + 1
            
            for team, count in team_counts.items():
                if count > self.MAX_PER_TEAM:
                    validation['is_valid'] = False
                    validation['violations'].append(f"Team {team}: {count} players (max: {self.MAX_PER_TEAM})")
            
            # Check budget
            total_cost = sum([p['cost'] for p in squad])
            if total_cost > self.BUDGET:
                validation['is_valid'] = False
                validation['violations'].append(f"Budget exceeded: £{total_cost/10:.1f}m (max: £{self.BUDGET/10:.1f}m)")
            
            # Summary
            validation['summary'] = {
                'squad_size': len(squad),
                'total_cost': total_cost / 10,
                'position_counts': {self.POSITIONS[pid]['name']: count for pid, count in position_counts.items()},
                'team_counts': team_counts,
                'budget_remaining': (self.BUDGET - total_cost) / 10
            }
            
            return validation
            
        except Exception as e:
            logger.error(f"Error validating squad: {e}")
            return {'is_valid': False, 'violations': ['Validation error'], 'summary': {}}
        
    def test_actual_optimization(self, predictions: Dict[int, Dict]) -> Dict:
        """Test the actual optimization process step by step."""
        try:
            logger.info("Testing actual optimization process...")
        
            # Get available players (same as optimization)
            players_df = self._get_available_players(predictions)
        
            if players_df.empty:
                return {'error': 'No available players found'}
        
            logger.info(f"Got {len(players_df)} players for optimization")
            logger.info(f"Position distribution: {players_df['position'].value_counts().to_dict()}")
        
            # Create optimization problem (same as real optimization)
            prob = LpProblem("FPL_Squad_Optimization_Test", LpMaximize)
        
            # Decision variables
            player_vars = {}
            for _, player in players_df.iterrows():
                player_vars[player['id']] = LpVariable(
                    f"player_{player['id']}", 
                    cat='Binary'
                )
        
            logger.info(f"Created {len(player_vars)} decision variables")
        
            # Objective function
            prob += lpSum([
                predictions[player_id]['points'] * player_vars[player_id] 
                for player_id in player_vars
                if player_id in predictions
            ]), "Total_Predicted_Points"
        
            logger.info("Added objective function")
        
            # Add constraints (same method as real optimization)
            self._add_squad_constraints(prob, player_vars, players_df, self.BUDGET)
        
            logger.info("Added all constraints")
        
            # Try to solve
            logger.info("Attempting to solve optimization problem...")
            prob.solve(PULP_CBC_CMD(msg=1))  # msg=1 for verbose output
        
            result = {
                'status': LpStatus[prob.status],
                'status_code': prob.status,
                'num_variables': len(player_vars),
                'num_constraints': len(prob.constraints),
                'objective_value': prob.objective.value() if prob.status == 1 else None
            }
        
            if prob.status == 1:  # Optimal
                selected_count = sum([1 for var in player_vars.values() if var.varValue == 1])
                result['selected_players'] = selected_count
                logger.info(f"✅ Optimization successful! Selected {selected_count} players")
            else:
                logger.error(f"❌ Optimization failed with status: {LpStatus[prob.status]}")
            
                # Try to diagnose the issue
                result['constraint_analysis'] = {}
                for name, constraint in prob.constraints.items():
                    result['constraint_analysis'][name] = str(constraint)
        
            return result
        
        except Exception as e:
            logger.error(f"Error in test optimization: {e}")
            return {'error': str(e)}

