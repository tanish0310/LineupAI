import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
import os
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

class TransferHitAnalyzer:
    """
    Advanced analysis system for determining when transfer hits (-4 points) are worthwhile.
    Considers immediate returns, future value, and strategic positioning.
    """
    
    def __init__(self):
        self.HIT_COST = 4  # Standard FPL transfer hit cost
        self.MIN_IMMEDIATE_RETURN = 1.0  # Minimum immediate return to consider a hit
        self.STRATEGIC_THRESHOLD = 2.0  # Threshold for strategic hits
        
    def analyze_hit_value(self, transfer_options: List[Dict], free_transfers: int,
                         future_predictions: Dict = None) -> List[Dict]:
        """
        Comprehensive analysis of transfer hit value across multiple scenarios.
        
        Args:
            transfer_options: List of transfer combinations
            free_transfers: Available free transfers
            future_predictions: Multi-gameweek predictions for strategic analysis
            
        Returns:
            Detailed hit analysis with recommendations
        """
        try:
            logger.info("Analyzing transfer hit values...")
            
            hit_analyses = []
            
            for option in transfer_options:
                if option['num_transfers'] > free_transfers:
                    # This transfer requires hits
                    hits_needed = option['num_transfers'] - free_transfers
                    hit_cost = hits_needed * self.HIT_COST
                    
                    analysis = {
                        'transfer_option': option,
                        'hits_needed': hits_needed,
                        'hit_cost': hit_cost,
                        'immediate_analysis': {},
                        'strategic_analysis': {},
                        'risk_assessment': {},
                        'recommendation': {}
                    }
                    
                    # Immediate return analysis
                    analysis['immediate_analysis'] = self._analyze_immediate_returns(
                        option, hit_cost
                    )
                    
                    # Strategic value analysis
                    if future_predictions:
                        analysis['strategic_analysis'] = self._analyze_strategic_value(
                            option, hit_cost, future_predictions
                        )
                    
                    # Risk assessment
                    analysis['risk_assessment'] = self._assess_hit_risks(option)
                    
                    # Generate final recommendation
                    analysis['recommendation'] = self._generate_hit_recommendation(
                        analysis['immediate_analysis'],
                        analysis.get('strategic_analysis', {}),
                        analysis['risk_assessment']
                    )
                    
                    hit_analyses.append(analysis)
            
            # Sort by recommendation quality
            hit_analyses.sort(
                key=lambda x: x['recommendation']['confidence_score'],
                reverse=True
            )
            
            logger.info(f"Analyzed {len(hit_analyses)} hit scenarios")
            return hit_analyses
            
        except Exception as e:
            logger.error(f"Error analyzing hit values: {e}")
            return []
    
    def _analyze_immediate_returns(self, transfer_option: Dict, hit_cost: int) -> Dict:
        """Analyze immediate gameweek returns from the transfer."""
        try:
            points_improvement = transfer_option['points_improvement']
            net_immediate_return = points_improvement - hit_cost
            
            analysis = {
                'points_improvement': points_improvement,
                'hit_cost': hit_cost,
                'net_return': net_immediate_return,
                'return_ratio': points_improvement / hit_cost if hit_cost > 0 else 0,
                'assessment': '',
                'confidence': transfer_option.get('confidence', 0.5)
            }
            
            # Assess immediate return quality
            if net_immediate_return >= 2.0:
                analysis['assessment'] = 'Excellent immediate return - strongly recommended'
            elif net_immediate_return >= 1.0:
                analysis['assessment'] = 'Good immediate return - recommended'
            elif net_immediate_return >= 0:
                analysis['assessment'] = 'Marginal immediate return - consider carefully'
            else:
                analysis['assessment'] = 'Negative immediate return - likely not worth it'
            
            # Risk-adjusted return
            analysis['risk_adjusted_return'] = net_immediate_return * analysis['confidence']
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing immediate returns: {e}")
            return {}
    
    def _analyze_strategic_value(self, transfer_option: Dict, hit_cost: int,
                               future_predictions: Dict) -> Dict:
        """Analyze multi-gameweek strategic value of the hit."""
        try:
            analysis = {
                'multi_gw_value': 0,
                'payback_period': 0,
                'long_term_benefit': 0,
                'strategic_factors': []
            }
            
            # Calculate multi-gameweek value (next 5 gameweeks)
            total_improvement = 0
            gameweeks_analyzed = min(5, len(future_predictions))
            
            for gw in range(1, gameweeks_analyzed + 1):
                gw_predictions = future_predictions.get(gw, {})
                
                # Calculate improvement for this gameweek
                gw_improvement = 0
                for player_out, player_in in zip(transfer_option.get('players_out', []), 
                                                transfer_option.get('players_in', [])):
                    points_out = gw_predictions.get(player_out['id'], {}).get('points', 0)
                    points_in = gw_predictions.get(player_in['id'], {}).get('points', 0)
                    gw_improvement += points_in - points_out
                
                total_improvement += gw_improvement
                
                # Calculate payback period
                if total_improvement >= hit_cost and analysis['payback_period'] == 0:
                    analysis['payback_period'] = gw
            
            analysis['multi_gw_value'] = total_improvement
            analysis['long_term_benefit'] = total_improvement - hit_cost
            
            # If no payback period found, estimate based on average improvement
            if analysis['payback_period'] == 0 and gameweeks_analyzed > 0:
                avg_improvement = total_improvement / gameweeks_analyzed
                if avg_improvement > 0:
                    analysis['payback_period'] = hit_cost / avg_improvement
                else:
                    analysis['payback_period'] = float('inf')
            
            # Identify strategic factors
            if analysis['payback_period'] <= 2:
                analysis['strategic_factors'].append('Quick payback within 2 gameweeks')
            
            if analysis['long_term_benefit'] > hit_cost:
                analysis['strategic_factors'].append(
                    f'Strong long-term value: +{analysis["long_term_benefit"]:.1f} points over {gameweeks_analyzed}GW'
                )
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing strategic value: {e}")
            return {}
    
    def _assess_hit_risks(self, transfer_option: Dict) -> Dict:
        """Assess risks associated with taking the hit."""
        try:
            risks = {
                'injury_risk': 0,
                'rotation_risk': 0,
                'form_risk': 0,
                'price_risk': 0,
                'overall_risk': 'Low',
                'risk_factors': []
            }
            
            # Assess risks for incoming players
            for player_in in transfer_option.get('players_in', []):
                # Injury risk
                if player_in.get('injury_risk', 0) > 0.3:
                    risks['injury_risk'] += 0.3
                    risks['risk_factors'].append(f"{player_in['name']} has injury concerns")
                
                # Rotation risk
                if player_in.get('rotation_risk', 0) > 0.4:
                    risks['rotation_risk'] += 0.25
                    risks['risk_factors'].append(f"{player_in['name']} faces rotation risk")
                
                # Form sustainability risk
                recent_form = player_in.get('form', 0)
                if recent_form > 5.0:  # Potentially unsustainable form
                    risks['form_risk'] += 0.2
                    risks['risk_factors'].append(f"{player_in['name']} may have unsustainable form")
                
                # Price rise risk (if player is rising fast)
                if player_in.get('price_momentum', 0) > 0.2:
                    risks['price_risk'] += 0.1
                    risks['risk_factors'].append(f"{player_in['name']} price may continue rising")
            
            # Calculate overall risk level
            total_risk = (risks['injury_risk'] + risks['rotation_risk'] + 
                         risks['form_risk'] + risks['price_risk'])
            
            if total_risk >= 0.6:
                risks['overall_risk'] = 'High'
            elif total_risk >= 0.3:
                risks['overall_risk'] = 'Medium'
            else:
                risks['overall_risk'] = 'Low'
            
            return risks
            
        except Exception as e:
            logger.error(f"Error assessing hit risks: {e}")
            return {}
    
    def _generate_hit_recommendation(self, immediate_analysis: Dict, 
                                   strategic_analysis: Dict, risk_assessment: Dict) -> Dict:
        """Generate comprehensive hit recommendation."""
        try:
            recommendation = {
                'action': 'AVOID',
                'confidence_score': 0.3,
                'reasoning': [],
                'summary': '',
                'alternative_suggestions': []
            }
            
            immediate_return = immediate_analysis.get('net_return', 0)
            long_term_benefit = strategic_analysis.get('long_term_benefit', 0)
            payback_period = strategic_analysis.get('payback_period', float('inf'))
            overall_risk = risk_assessment.get('overall_risk', 'High')
            
            # Decision logic
            confidence_factors = []
            
            # Strong immediate return
            if immediate_return >= 2:
                recommendation['action'] = 'TAKE HIT'
                recommendation['reasoning'].append(f'Strong immediate return: +{immediate_return:.1f} net points')
                confidence_factors.append(0.3)
            elif immediate_return >= 1:
                recommendation['action'] = 'CONSIDER'
                recommendation['reasoning'].append(f'Good immediate return: +{immediate_return:.1f} net points')
                confidence_factors.append(0.2)
            elif immediate_return >= 0:
                recommendation['reasoning'].append(f'Marginal immediate return: +{immediate_return:.1f} net points')
                confidence_factors.append(0.1)
            else:
                recommendation['reasoning'].append(f'Negative immediate return: {immediate_return:.1f} points')
                confidence_factors.append(-0.2)
            
            # Strategic value consideration
            if strategic_analysis:
                if long_term_benefit > 4:
                    recommendation['action'] = 'TAKE HIT'
                    recommendation['reasoning'].append(f'Excellent long-term value: +{long_term_benefit:.1f} points')
                    confidence_factors.append(0.2)
                elif long_term_benefit > 2:
                    if recommendation['action'] == 'AVOID':
                        recommendation['action'] = 'CONSIDER'
                    recommendation['reasoning'].append(f'Good long-term value: +{long_term_benefit:.1f} points')
                    confidence_factors.append(0.15)
                elif payback_period <= 3:
                    recommendation['reasoning'].append(f'Reasonable payback period: {payback_period:.1f} gameweeks')
                    confidence_factors.append(0.1)
            
            # Risk adjustment
            if overall_risk == 'High':
                recommendation['reasoning'].append('High risk factors present')
                confidence_factors.append(-0.15)
            elif overall_risk == 'Low':
                confidence_factors.append(0.1)
            
            # Calculate confidence score
            base_confidence = 0.5
            confidence_adjustment = sum(confidence_factors)
            recommendation['confidence_score'] = max(0.1, min(0.95, base_confidence + confidence_adjustment))
            
            # Refine action based on confidence
            if recommendation['confidence_score'] >= 0.7:
                if recommendation['action'] == 'CONSIDER':
                    recommendation['action'] = 'TAKE HIT'
                elif recommendation['action'] == 'AVOID' and immediate_return >= 0:
                    recommendation['action'] = 'CONSIDER'
            elif recommendation['confidence_score'] < 0.4:
                recommendation['action'] = 'AVOID'
            
            # Generate summary
            recommendation['summary'] = f"{recommendation['action']} (Confidence: {recommendation['confidence_score']:.1f})"
            
            # Add alternative suggestions
            if recommendation['action'] == 'AVOID':
                recommendation['alternative_suggestions'] = [
                    'Wait for next gameweek with free transfers',
                    'Consider taking just 1 hit for the best player',
                    'Look for similar players at lower cost'
                ]
            elif recommendation['action'] == 'CONSIDER':
                recommendation['alternative_suggestions'] = [
                    'Monitor injury news before deadline',
                    'Consider if you have other pressing transfers',
                    'Evaluate fixture swing timing'
                ]
            
            return recommendation
            
        except Exception as e:
            logger.error(f"Error generating hit recommendation: {e}")
            return {
                'action': 'AVOID',
                'confidence_score': 0.3,
                'reasoning': ['Analysis error'],
                'summary': 'Unable to analyze - avoid hit',
                'alternative_suggestions': []
            }
    
    def compare_hit_scenarios(self, hit_analyses: List[Dict]) -> Dict:
        """Compare multiple hit scenarios and provide ranking."""
        try:
            if not hit_analyses:
                return {'ranking': [], 'summary': 'No hit scenarios to analyze'}
            
            # Sort by combined immediate + strategic value
            for analysis in hit_analyses:
                immediate_value = analysis['immediate_analysis'].get('net_return', 0)
                strategic_value = analysis.get('strategic_analysis', {}).get('long_term_benefit', 0)
                risk_penalty = -1 if analysis['risk_assessment']['overall_risk'] == 'High' else 0
                
                analysis['combined_score'] = immediate_value + strategic_value * 0.3 + risk_penalty
            
            # Sort by combined score
            hit_analyses.sort(key=lambda x: x['combined_score'], reverse=True)
            
            # Create ranking
            ranking = []
            for i, analysis in enumerate(hit_analyses[:5]):  # Top 5
                summary = analysis['transfer_option']['transfer_summary'] if 'transfer_summary' in analysis['transfer_option'] else f"Option {i+1}"
                
                ranking.append({
                    'rank': i + 1,
                    'transfer': summary,
                    'action': analysis['recommendation']['action'],
                    'immediate_return': analysis['immediate_analysis']['net_return'],
                    'combined_score': analysis['combined_score'],
                    'confidence': analysis['recommendation']['confidence_score'],
                    'summary': analysis['recommendation']['summary']
                })
            
            # Generate overall summary
            best_option = ranking[0] if ranking else None
            if best_option and best_option['action'] == 'TAKE HIT':
                summary = f"Best hit option: {best_option['transfer']} ({best_option['immediate_return']:+.1f} net points)"
            else:
                summary = "No hits currently recommended - wait for free transfers"
            
            return {
                'ranking': ranking,
                'summary': summary,
                'best_option': best_option
            }
            
        except Exception as e:
            logger.error(f"Error comparing hit scenarios: {e}")
            return {'ranking': [], 'summary': 'Analysis error'}
    
    def generate_hit_timing_advice(self, transfer_options: List[Dict], 
                                 upcoming_fixtures: Dict) -> Dict:
        """Generate advice on optimal timing for taking hits."""
        try:
            timing_advice = {
                'current_gameweek': {'recommended': False, 'reasoning': []},
                'next_gameweek': {'recommended': False, 'reasoning': []},
                'future_consideration': {'recommended': False, 'reasoning': []},
                'overall_recommendation': ''
            }
            
            # Analyze current gameweek hits
            current_gw_value = 0
            for option in transfer_options:
                if option.get('net_improvement', 0) > 1:
                    current_gw_value += option['net_improvement']
            
            if current_gw_value >= 3:
                timing_advice['current_gameweek']['recommended'] = True
                timing_advice['current_gameweek']['reasoning'].append(
                    f'Strong immediate value (+{current_gw_value:.1f} net points)'
                )
            
            # Consider fixture-based timing
            if upcoming_fixtures:
                # This would analyze upcoming fixture difficulty changes
                # For now, providing framework
                timing_advice['future_consideration']['reasoning'].append(
                    'Consider fixture swings in gameweeks 3-5'
                )
            
            # Generate overall recommendation
            if timing_advice['current_gameweek']['recommended']:
                timing_advice['overall_recommendation'] = 'Take hits now for immediate value'
            else:
                timing_advice['overall_recommendation'] = 'Wait for better opportunities or free transfers'
            
            return timing_advice
            
        except Exception as e:
            logger.error(f"Error generating timing advice: {e}")
            return {}

