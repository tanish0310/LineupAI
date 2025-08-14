import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
from typing import Dict, List, Tuple, Optional
import pickle
import logging
import os
from datetime import datetime
from sqlalchemy import create_engine
from dotenv import load_dotenv

from models.prediction.feature_engineer import FeatureEngineer

load_dotenv()

logger = logging.getLogger(__name__)

class PlayerPredictor:
    """
    Advanced position-specific ML prediction system for FPL players.
    Uses different models optimized for each position's unique characteristics.
    """
    
    def __init__(self):
        self.engine = create_engine(os.getenv('DATABASE_URL'))
        self.feature_engineer = FeatureEngineer()
        
        # Position-specific models
        self.models = {
            'goalkeeper': None,
            'defender': None,
            'midfielder': None,
            'forward': None
        }
        
        # Feature scalers for each position
        self.scalers = {
            'goalkeeper': StandardScaler(),
            'defender': StandardScaler(),
            'midfielder': StandardScaler(),
            'forward': StandardScaler()
        }
        
        # Model configurations for each position
        self.model_configs = {
            'goalkeeper': {
                'model_type': 'xgboost',
                'params': {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'random_state': 42
                }
            },
            'defender': {
                'model_type': 'random_forest',
                'params': {
                    'n_estimators': 150,
                    'max_depth': 12,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'random_state': 42
                }
            },
            'midfielder': {
                'model_type': 'xgboost',
                'params': {
                    'n_estimators': 200,
                    'max_depth': 8,
                    'learning_rate': 0.05,
                    'subsample': 0.9,
                    'colsample_bytree': 0.9,
                    'random_state': 42
                }
            },
            'forward': {
                'model_type': 'xgboost',
                'params': {
                    'n_estimators': 150,
                    'max_depth': 7,
                    'learning_rate': 0.08,
                    'subsample': 0.85,
                    'colsample_bytree': 0.85,
                    'random_state': 42
                }
            }
        }
        
        # Feature importance tracking
        self.feature_importance = {}
        
        # Model performance metrics
        self.performance_metrics = {}
    
    def prepare_training_data(self, seasons: List[str] = ['2022-23', '2023-24']) -> Dict[str, pd.DataFrame]:
        """
        Prepare position-specific training datasets from historical data.
        """
        try:
            logger.info("Preparing training data for all positions...")
            
            training_data = {}
            
            for position_id, position_name in self.feature_engineer.position_configs.items():
                logger.info(f"Preparing training data for {position_name}...")
                
                # Get historical data for this position
                position_data = self._get_historical_position_data(position_id, seasons)
                
                if position_data.empty:
                    logger.warning(f"No training data found for {position_name}")
                    continue
                
                # Engineer features for historical data
                features_data = self._engineer_historical_features(position_data, position_id)
                
                if features_data.empty:
                    logger.warning(f"No features generated for {position_name}")
                    continue
                
                # Prepare final training dataset
                training_dataset = self._prepare_final_training_data(features_data, position_name)
                
                training_data[position_name] = training_dataset
                
                logger.info(f"Training data prepared for {position_name}: {len(training_dataset)} samples")
            
            return training_data
            
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            raise
    
    def _get_historical_position_data(self, position_id: int, seasons: List[str]) -> pd.DataFrame:
        """Get historical gameweek data for specific position."""
        try:
            # Get gameweeks from specified seasons
            season_filter = " OR ".join([f"gw.name LIKE '%{season}%'" for season in seasons])
            
            query = f"""
            SELECT 
                gs.*,
                p.position,
                p.team,
                p.now_cost,
                p.web_name,
                gw.name as gameweek_name,
                gw.deadline_time
            FROM gameweek_stats gs
            JOIN players p ON gs.player_id = p.id
            JOIN gameweeks gw ON gs.gameweek_id = gw.id
            WHERE p.position = %s
            AND gs.minutes > 0
            AND ({season_filter})
            ORDER BY gs.player_id, gs.gameweek_id
            """

            data = pd.read_sql(query, self.engine, params=[position_id])

            
            
            return data
            
        except Exception as e:
            logger.error(f"Error getting historical data for position {position_id}: {e}")
            return pd.DataFrame()
    
    def _engineer_historical_features(self, position_data: pd.DataFrame, position_id: int) -> pd.DataFrame:
        """Engineer features for historical data."""
        try:
            features_list = []
            
            # Group by player to maintain time series order
            for player_id in position_data['player_id'].unique():
                player_data = position_data[position_data['player_id'] == player_id].copy()
                player_data = player_data.sort_values('gameweek_id')
                
                # Calculate rolling features for each gameweek
                for idx, row in player_data.iterrows():
                    gameweek_id = row['gameweek_id']
                    
                    # Get historical data up to this point
                    historical_data = player_data[player_data['gameweek_id'] < gameweek_id]
                    
                    if len(historical_data) < 3:  # Need minimum history
                        continue
                    
                    # Create feature set based on position
                    position_name = self.feature_engineer.position_configs[position_id]
                    
                    if position_name == 'goalkeeper':
                        features = self._create_gk_historical_features(historical_data, row)
                    elif position_name == 'defender':
                        features = self._create_def_historical_features(historical_data, row)
                    elif position_name == 'midfielder':
                        features = self._create_mid_historical_features(historical_data, row)
                    elif position_name == 'forward':
                        features = self._create_fwd_historical_features(historical_data, row)
                    
                    # Add target variable (points scored in this gameweek)
                    features['target_points'] = row['total_points']
                    features['player_id'] = player_id
                    features['gameweek_id'] = gameweek_id
                    
                    features_list.append(features)
            
            return pd.DataFrame(features_list)
            
        except Exception as e:
            logger.error(f"Error engineering historical features: {e}")
            return pd.DataFrame()
    
    def _create_gk_historical_features(self, historical_data: pd.DataFrame, current_row: pd.Series) -> Dict:
        """Create goalkeeper-specific features from historical data."""
        features = {}
        
        # Rolling averages for different periods
        for period in [3, 5, 10]:
            recent = historical_data.tail(period)
            if len(recent) > 0:
                features[f'avg_points_{period}gw'] = recent['total_points'].mean()
                features[f'avg_saves_{period}gw'] = recent['saves'].mean()
                features[f'avg_clean_sheets_{period}gw'] = recent['clean_sheets'].mean()
                features[f'avg_minutes_{period}gw'] = recent['minutes'].mean()
        
        # Form metrics
        if len(historical_data) >= 5:
            features['points_trend'] = historical_data.tail(5)['total_points'].mean() - historical_data.head(5)['total_points'].mean()
            features['consistency'] = 1 / (1 + historical_data['total_points'].std())
        
        # Goalkeeper-specific metrics
        features['save_rate'] = historical_data['saves'].mean()
        features['clean_sheet_rate'] = historical_data['clean_sheets'].mean()
        features['goals_conceded_rate'] = historical_data['goals_conceded'].mean()
        
        # Current gameweek context
        features['current_cost'] = current_row['now_cost']
        features['position'] = 1
        
        return features
    
    def _create_def_historical_features(self, historical_data: pd.DataFrame, current_row: pd.Series) -> Dict:
        """Create defender-specific features from historical data."""
        features = {}
        
        # Rolling averages
        for period in [3, 5, 10]:
            recent = historical_data.tail(period)
            if len(recent) > 0:
                features[f'avg_points_{period}gw'] = recent['total_points'].mean()
                features[f'avg_clean_sheets_{period}gw'] = recent['clean_sheets'].mean()
                features[f'avg_goal_involvements_{period}gw'] = (recent['goals_scored'] + recent['assists']).mean()
                features[f'avg_bonus_{period}gw'] = recent['bonus'].mean()
                features[f'avg_bps_{period}gw'] = recent['bps'].mean()
        
        # Defender-specific metrics
        features['clean_sheet_rate'] = historical_data['clean_sheets'].mean()
        features['attacking_threat'] = (historical_data['goals_scored'] + historical_data['assists']).mean()
        features['bonus_frequency'] = (historical_data['bonus'] > 0).mean()
        
        # Form and consistency
        if len(historical_data) >= 5:
            features['form_trend'] = historical_data.tail(5)['total_points'].mean() - historical_data.head(5)['total_points'].mean()
            features['consistency'] = 1 / (1 + historical_data['total_points'].std())
        
        features['current_cost'] = current_row['now_cost']
        features['position'] = 2
        
        return features
    
    def _create_mid_historical_features(self, historical_data: pd.DataFrame, current_row: pd.Series) -> Dict:
        """Create midfielder-specific features from historical data."""
        features = {}
        
        # Rolling averages
        for period in [3, 5, 10]:
            recent = historical_data.tail(period)
            if len(recent) > 0:
                features[f'avg_points_{period}gw'] = recent['total_points'].mean()
                features[f'avg_goals_{period}gw'] = recent['goals_scored'].mean()
                features[f'avg_assists_{period}gw'] = recent['assists'].mean()
                features[f'avg_creativity_{period}gw'] = recent['creativity'].mean()
                features[f'avg_ict_{period}gw'] = recent['ict_index'].mean()
        
        # Midfielder-specific metrics
        features['goals_per_game'] = historical_data['goals_scored'].mean()
        features['assists_per_game'] = historical_data['assists'].mean()
        features['creativity_index'] = historical_data['creativity'].mean()
        features['ict_involvement'] = historical_data['ict_index'].mean()
        
        # Attack involvement
        features['goal_involvements_per_game'] = (historical_data['goals_scored'] + historical_data['assists']).mean()
        
        # Form metrics
        if len(historical_data) >= 5:
            features['form_trend'] = historical_data.tail(5)['total_points'].mean() - historical_data.head(5)['total_points'].mean()
            features['consistency'] = 1 / (1 + historical_data['total_points'].std())
        
        features['current_cost'] = current_row['now_cost']
        features['position'] = 3
        
        return features
    
    def _create_fwd_historical_features(self, historical_data: pd.DataFrame, current_row: pd.Series) -> Dict:
        """Create forward-specific features from historical data."""
        features = {}
        
        # Rolling averages
        for period in [3, 5, 10]:
            recent = historical_data.tail(period)
            if len(recent) > 0:
                features[f'avg_points_{period}gw'] = recent['total_points'].mean()
                features[f'avg_goals_{period}gw'] = recent['goals_scored'].mean()
                features[f'avg_assists_{period}gw'] = recent['assists'].mean()
                features[f'avg_threat_{period}gw'] = recent['threat'].mean()
        
        # Forward-specific metrics
        features['goals_per_game'] = historical_data['goals_scored'].mean()
        features['assists_per_game'] = historical_data['assists'].mean()
        features['threat_index'] = historical_data['threat'].mean()
        
        # Shooting metrics (approximated)
        features['shots_conversion'] = historical_data['goals_scored'].sum() / max(historical_data['threat'].sum() / 10, 1)
        
        # Minutes reliability
        features['minutes_reliability'] = (historical_data['minutes'] >= 60).mean()
        
        # Form metrics
        if len(historical_data) >= 5:
            features['form_trend'] = historical_data.tail(5)['total_points'].mean() - historical_data.head(5)['total_points'].mean()
            features['consistency'] = 1 / (1 + historical_data['total_points'].std())
        
        features['current_cost'] = current_row['now_cost']
        features['position'] = 4
        
        return features
    
    def _prepare_final_training_data(self, features_data: pd.DataFrame, position_name: str) -> pd.DataFrame:
        """Prepare final training dataset with proper feature selection and cleaning."""
        try:
            if features_data.empty:
                return pd.DataFrame()
            
            # Remove rows with missing target
            features_data = features_data.dropna(subset=['target_points'])
            
            # Fill missing feature values
            numeric_columns = features_data.select_dtypes(include=[np.number]).columns
            features_data[numeric_columns] = features_data[numeric_columns].fillna(0)
            
            # Remove outliers (points > 25 are very rare and can skew model)
            features_data = features_data[features_data['target_points'] <= 25]
            
            # Sort by gameweek for time series split
            features_data = features_data.sort_values(['gameweek_id', 'player_id'])
            
            return features_data
            
        except Exception as e:
            logger.error(f"Error preparing final training data for {position_name}: {e}")
            return pd.DataFrame()
    
    def train_position_models(self, training_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """
        Train position-specific models using historical data with time series validation.
        """
        try:
            logger.info("Training position-specific models...")
            
            results = {}
            
            for position_name, data in training_data.items():
                if data.empty:
                    logger.warning(f"No training data for {position_name}, skipping...")
                    continue
                
                logger.info(f"Training {position_name} model with {len(data)} samples...")
                
                # Prepare features and target
                feature_columns = [col for col in data.columns if col not in ['target_points', 'player_id', 'gameweek_id']]
                X = data[feature_columns].copy()
                y = data['target_points'].copy()
                
                # Handle any remaining missing values
                X = X.fillna(0)
                
                # Scale features
                X_scaled = self.scalers[position_name].fit_transform(X)
                X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
                
                # Train with time series cross-validation
                model_result = self._train_single_position_model(
                    X_scaled, y, position_name, feature_columns
                )
                
                results[position_name] = model_result
                
                logger.info(f"Completed training for {position_name}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error training position models: {e}")
            raise
    
    def _train_single_position_model(self, X: pd.DataFrame, y: pd.Series, 
                                   position_name: str, feature_columns: List[str]) -> Dict:
        """Train a single position-specific model with cross-validation."""
        try:
            config = self.model_configs[position_name]
            
            # Initialize model based on configuration
            if config['model_type'] == 'xgboost':
                model = xgb.XGBRegressor(**config['params'])
            elif config['model_type'] == 'random_forest':
                model = RandomForestRegressor(**config['params'])
            else:
                raise ValueError(f"Unknown model type: {config['model_type']}")
            
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=5)
            cv_scores = []
            
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Train model on fold
                model.fit(X_train, y_train)
                
                # Validate
                y_pred = model.predict(X_val)
                mae = mean_absolute_error(y_val, y_pred)
                rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                
                cv_scores.append({'mae': mae, 'rmse': rmse})
                
                logger.info(f"{position_name} Fold {fold+1}: MAE={mae:.3f}, RMSE={rmse:.3f}")
            
            # Train final model on all data
            model.fit(X, y)
            
            # Store model
            self.models[position_name] = model
            
            # Calculate feature importance
            if hasattr(model, 'feature_importances_'):
                feature_importance = dict(zip(feature_columns, model.feature_importances_))
                self.feature_importance[position_name] = feature_importance
            
            # Store performance metrics
            avg_mae = np.mean([score['mae'] for score in cv_scores])
            avg_rmse = np.mean([score['rmse'] for score in cv_scores])
            
            self.performance_metrics[position_name] = {
                'cv_mae': avg_mae,
                'cv_rmse': avg_rmse,
                'cv_scores': cv_scores,
                'n_features': len(feature_columns),
                'n_samples': len(X)
            }
            
            # Save model to disk
            self._save_model(model, position_name)
            
            return {
                'model': model,
                'performance': self.performance_metrics[position_name],
                'feature_importance': self.feature_importance.get(position_name, {}),
                'feature_columns': feature_columns
            }
            
        except Exception as e:
            logger.error(f"Error training {position_name} model: {e}")
            raise

    def train_position_models(self, training_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """Train position-specific models using historical data with time series validation."""
        try:
            logger.info("Training position-specific models...")
        
            results = {}
        
            for position_name, data in training_data.items():
                if data.empty:
                    logger.warning(f"No training data for {position_name}, skipping...")
                    continue
            
                logger.info(f"Training {position_name} model with {len(data)} samples...")
            
                # Check for minimum data requirement
                if len(data) < 10:
                    logger.warning(f"Insufficient data for {position_name}: {len(data)} samples. Creating simple model for testing.")
                    # Create a simple fallback model for testing
                    from sklearn.linear_model import LinearRegression
                    self.models[position_name] = LinearRegression()
                    continue
            
                # Prepare features and target
                feature_columns = [col for col in data.columns if col not in ['target_points', 'player_id', 'gameweek_id']]
                X = data[feature_columns].copy()
                y = data['target_points'].copy()
            
                # Handle any remaining missing values
                X = X.fillna(0)
            
                # Scale features
                X_scaled = self.scalers[position_name].fit_transform(X)
                X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
            
                # Train with time series cross-validation
                model_result = self._train_single_position_model(
                    X_scaled, y, position_name, feature_columns
                )
            
                results[position_name] = model_result
            
                logger.info(f"Completed training for {position_name}")
        
            return results
        
        except Exception as e:
            logger.error(f"Error training position models: {e}")
            raise

    
    def predict_gameweek_points(self, gameweek_id: int) -> Dict[int, Dict]:
        """
        Generate predictions for all players for the specified gameweek.
        Returns predictions with confidence scores.
        """
        try:
            logger.info(f"Generating predictions for gameweek {gameweek_id}...")
            
            predictions = {}
            
            # Get all available players
            players_query = """
            SELECT id, position, team, web_name, now_cost, total_points, form, status
            FROM players 
            WHERE status = 'a'
            ORDER BY position, total_points DESC
            """
            
            players_df = pd.read_sql(players_query, self.engine)
            
            for position_id in [1, 2, 3, 4]:  # All positions
                position_name = self.feature_engineer.position_configs[position_id]
                position_players = players_df[players_df['position'] == position_id]
                
                if position_players.empty:
                    continue
                
                if self.models[position_name] is None:
                    logger.warning(f"No trained model for {position_name}, loading from disk...")
                    self._load_model(position_name)
                
                if self.models[position_name] is None:
                    logger.error(f"No model available for {position_name}")
                    continue
                
                # Generate predictions for this position
                position_predictions = self._predict_position_players(
                    position_players, gameweek_id, position_name
                )
                
                predictions.update(position_predictions)
                
                logger.info(f"Generated {len(position_predictions)} predictions for {position_name}")
            
            # Store predictions in database
            self._store_predictions(predictions, gameweek_id)
            
            logger.info(f"Completed predictions for gameweek {gameweek_id}: {len(predictions)} players")
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error predicting gameweek points: {e}")
            raise
    
    def _predict_position_players(self, players_df: pd.DataFrame, gameweek_id: int, 
                                position_name: str) -> Dict[int, Dict]:
        """Generate predictions for all players in a specific position."""
        try:
            predictions = {}
            model = self.models[position_name]
            
            for _, player in players_df.iterrows():
                try:
                    # Generate features for this player
                    player_data = pd.DataFrame([player])
                    
                    if position_name == 'goalkeeper':
                        features = self.feature_engineer.goalkeeper_features(player_data, gameweek_id)
                    elif position_name == 'defender':
                        features = self.feature_engineer.defender_features(player_data, gameweek_id)
                    elif position_name == 'midfielder':
                        features = self.feature_engineer.midfielder_features(player_data, gameweek_id)
                    elif position_name == 'forward':
                        features = self.feature_engineer.forward_features(player_data, gameweek_id)
                    
                    if not features:
                        continue
                    
                    # Convert features to DataFrame for prediction
                    feature_df = pd.DataFrame([features])
                    
                    # Align features with training features
                    feature_df = self._align_prediction_features(feature_df, position_name)
                    
                    if feature_df.empty:
                        continue
                    
                    # Scale features
                    feature_scaled = self.scalers[position_name].transform(feature_df)
                    
                    # Make prediction
                    predicted_points = model.predict(feature_scaled)[0]
                    
                    # Calculate confidence score
                    confidence = self._calculate_confidence(features, position_name)
                    
                    # Apply position-specific adjustments
                    adjusted_prediction = self._apply_position_adjustments(
                        predicted_points, features, position_name
                    )
                    
                    predictions[player['id']] = {
                        'points': max(0, adjusted_prediction),  # Ensure non-negative
                        'confidence': confidence,
                        'raw_prediction': predicted_points,
                        'position': position_name,
                        'features_used': len(features)
                    }
                    
                except Exception as e:
                    logger.warning(f"Error predicting for player {player['id']}: {e}")
                    continue
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error predicting {position_name} players: {e}")
            return {}
    
    def _align_prediction_features(self, feature_df: pd.DataFrame, position_name: str) -> pd.DataFrame:
        """Align prediction features with training features."""
        try:
            # This would align features with the training set
            # For now, returning as-is with basic validation
            
            # Remove non-numeric columns
            numeric_df = feature_df.select_dtypes(include=[np.number])
            
            # Fill missing values
            numeric_df = numeric_df.fillna(0)
            
            return numeric_df
            
        except Exception as e:
            logger.error(f"Error aligning features for {position_name}: {e}")
            return pd.DataFrame()
    
    def _calculate_confidence(self, features: Dict, position_name: str) -> float:
        """Calculate prediction confidence based on feature quality and model performance."""
        try:
            base_confidence = 0.7  # Base confidence level
            
            # Adjust based on data quality
            availability_score = features.get('availability_score', 1.0)
            minutes_certainty = features.get('minutes_certainty', 0.8)
            
            # Adjust based on model performance
            model_performance = self.performance_metrics.get(position_name, {})
            mae = model_performance.get('cv_mae', 2.0)
            
            # Lower MAE = higher confidence
            performance_factor = max(0.5, 1.0 - (mae / 10))
            
            # Combine factors
            confidence = base_confidence * availability_score * minutes_certainty * performance_factor
            
            return max(0.1, min(1.0, confidence))
            
        except Exception:
            return 0.6  # Default confidence
    
    def _apply_position_adjustments(self, prediction: float, features: Dict, position_name: str) -> float:
        """Apply position-specific adjustments to raw predictions."""
        try:
            adjusted = prediction
            
            # Apply availability adjustments
            availability = features.get('availability_score', 1.0)
            adjusted *= availability
            
            # Position-specific adjustments
            if position_name == 'goalkeeper':
                # Goalkeepers have lower variance, cap at reasonable maximum
                adjusted = min(adjusted, 12.0)
            elif position_name == 'defender':
                # Defenders rarely score huge points
                adjusted = min(adjusted, 15.0)
            elif position_name == 'midfielder':
                # Midfielders can have highest upside
                pass  # No cap
            elif position_name == 'forward':
                # Forwards can have high variance
                pass  # No cap
            
            return adjusted
            
        except Exception:
            return prediction
    
    def _store_predictions(self, predictions: Dict[int, Dict], gameweek_id: int):
        """Store predictions in database."""
        try:
            # Prepare data for insertion
            prediction_records = []
            
            for player_id, pred_data in predictions.items():
                record = {
                    'player_id': player_id,
                    'gameweek_id': gameweek_id,
                    'predicted_points': pred_data['points'],
                    'confidence_score': pred_data['confidence'],
                    'prediction_method': f"{pred_data['position']}_model",
                    'created_at': datetime.now()
                }
                prediction_records.append(record)
            
            # Insert into database
            predictions_df = pd.DataFrame(prediction_records)
            predictions_df.to_sql('predictions', self.engine, if_exists='append', index=False)
            
            logger.info(f"Stored {len(prediction_records)} predictions in database")
            
        except Exception as e:
            logger.error(f"Error storing predictions: {e}")
    
    def _save_model(self, model, position_name: str):
        """Save trained model to disk."""
        try:
            model_dir = 'models/saved'
            os.makedirs(model_dir, exist_ok=True)
            
            model_path = f"{model_dir}/{position_name}_model.pkl"
            
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            # Save scaler as well
            scaler_path = f"{model_dir}/{position_name}_scaler.pkl"
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scalers[position_name], f)
            
            logger.info(f"Saved {position_name} model to {model_path}")
            
        except Exception as e:
            logger.error(f"Error saving {position_name} model: {e}")
    
    def _load_model(self, position_name: str):
        """Load trained model from disk."""
        try:
            model_path = f"models/saved/{position_name}_model.pkl"
            scaler_path = f"models/saved/{position_name}_scaler.pkl"
            
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    self.models[position_name] = pickle.load(f)
                
                if os.path.exists(scaler_path):
                    with open(scaler_path, 'rb') as f:
                        self.scalers[position_name] = pickle.load(f)
                
                logger.info(f"Loaded {position_name} model from disk")
            else:
                logger.warning(f"No saved model found for {position_name}")
                
        except Exception as e:
            logger.error(f"Error loading {position_name} model: {e}")
    
    def calculate_captain_multiplier(self, player_id: int, predicted_points: float, gameweek_id: int) -> Dict:
        """
        Calculate captaincy potential for a player including risk/reward analysis.
        """
        try:
            # Get player position and recent performance
            player_query = """
            SELECT position, web_name, total_points, form
            FROM players 
            WHERE id = %s
            """

            player_data = pd.read_sql(player_query, self.engine, params=[player_id])

            
            if player_data.empty:
                return {'captain_score': 0, 'reasoning': 'Player not found'}
            
            position = player_data['position'].iloc[0]
            
            # Base captain score (doubled points)
            base_captain_points = predicted_points * 2
            
            # Calculate safety factor based on consistency
            consistency_score = self._calculate_consistency_score(player_id)
            
            # Calculate fixture favorability
            fixture_score = self._calculate_fixture_favorability(player_id, gameweek_id)
            
            # Position-specific captain factors
            if position == 1:  # Goalkeeper
                safety_factor = 0.9  # Very safe but low ceiling
                upside_factor = 0.7
            elif position == 2:  # Defender
                safety_factor = 0.8  # Safe but limited upside
                upside_factor = 0.8
            elif position == 3:  # Midfielder
                safety_factor = 0.75  # Good balance
                upside_factor = 1.0
            elif position == 4:  # Forward
                safety_factor = 0.7  # Higher risk/reward
                upside_factor = 1.2
            
            # Calculate overall captain score
            captain_score = (
                base_captain_points * 
                (consistency_score * safety_factor + fixture_score * upside_factor) / 2
            )
            
            # Generate reasoning
            reasoning = self._generate_captain_reasoning(
                player_data, predicted_points, consistency_score, fixture_score, position
            )
            
            return {
                'captain_score': captain_score,
                'expected_points': base_captain_points,
                'safety_score': consistency_score,
                'fixture_score': fixture_score,
                'upside_potential': predicted_points * upside_factor,
                'reasoning': reasoning
            }
            
        except Exception as e:
            logger.error(f"Error calculating captain multiplier for player {player_id}: {e}")
            return {'captain_score': 0, 'reasoning': 'Error in calculation'}
    
    def _calculate_consistency_score(self, player_id: int) -> float:
        """Calculate player consistency based on recent performances."""
        try:
            query = """
            SELECT total_points
            FROM gameweek_stats
            WHERE player_id = %s
            AND gameweek_id > (SELECT MAX(id) - 10 FROM gameweeks WHERE finished = TRUE)
            ORDER BY gameweek_id DESC
            """
            recent_points = pd.read_sql(query, self.engine, params=[player_id])
            
            
            if len(recent_points) < 3:
                return 0.5
            
            # Calculate coefficient of variation (lower = more consistent)
            points_std = recent_points['total_points'].std()
            points_mean = recent_points['total_points'].mean()
            
            if points_mean == 0:
                return 0.3
            
            cv = points_std / points_mean
            consistency = max(0.2, 1.0 - min(cv, 1.0))
            
            return consistency
            
        except Exception:
            return 0.5
    
    def _calculate_fixture_favorability(self, player_id: int, gameweek_id: int) -> float:
        """Calculate fixture favorability for captaincy."""
        try:
            # Get fixture difficulty and opponent strength
            query = """
            SELECT 
                CASE 
                    WHEN p.team = f.team_h THEN f.team_h_difficulty
                    ELSE f.team_a_difficulty
                END as difficulty,
                p.team = f.team_h as is_home
            FROM players p
            JOIN fixtures f ON (p.team = f.team_h OR p.team = f.team_a)
            WHERE p.id = %s
            AND f.gameweek_id = %s
            """
            fixture_data = pd.read_sql(query, self.engine, params=[player_id, gameweek_id])
            
            if fixture_data.empty:
                return 0.5
            
            difficulty = fixture_data['difficulty'].iloc[0]
            is_home = fixture_data['is_home'].iloc[0]
            
            # Lower difficulty = better for captaincy
            difficulty_score = (6 - difficulty) / 4  # Scale 1-5 to 1.25-0.25
            
            # Home advantage
            home_bonus = 0.1 if is_home else 0
            
            fixture_favorability = difficulty_score + home_bonus
            
            return max(0.1, min(1.0, fixture_favorability))
            
        except Exception:
            return 0.5
    
    def _generate_captain_reasoning(self, player_data: pd.DataFrame, predicted_points: float,
                                  consistency_score: float, fixture_score: float, position: int) -> str:
        """Generate human-readable reasoning for captain recommendation."""
        try:
            name = player_data['web_name'].iloc[0]
            form = player_data['form'].iloc[0]
            
            reasoning_parts = []
            
            # Predicted points reasoning
            if predicted_points >= 8:
                reasoning_parts.append(f"High predicted points ({predicted_points:.1f})")
            elif predicted_points >= 6:
                reasoning_parts.append(f"Good predicted points ({predicted_points:.1f})")
            else:
                reasoning_parts.append(f"Moderate predicted points ({predicted_points:.1f})")
            
            # Form reasoning
            if form >= 4:
                reasoning_parts.append("excellent recent form")
            elif form >= 3:
                reasoning_parts.append("good recent form")
            else:
                reasoning_parts.append("mixed recent form")
            
            # Consistency reasoning
            if consistency_score >= 0.8:
                reasoning_parts.append("very consistent performer")
            elif consistency_score >= 0.6:
                reasoning_parts.append("fairly consistent")
            else:
                reasoning_parts.append("inconsistent but high upside")
            
            # Fixture reasoning
            if fixture_score >= 0.8:
                reasoning_parts.append("excellent fixture")
            elif fixture_score >= 0.6:
                reasoning_parts.append("favorable fixture")
            else:
                reasoning_parts.append("challenging fixture")
            
            return f"{name}: " + ", ".join(reasoning_parts)
            
        except Exception:
            return "Standard captain recommendation"
    
    def get_model_performance_summary(self) -> Dict:
        """Get summary of all model performances."""
        try:
            summary = {}
            
            for position, metrics in self.performance_metrics.items():
                summary[position] = {
                    'mae': metrics.get('cv_mae', 0),
                    'rmse': metrics.get('cv_rmse', 0),
                    'samples': metrics.get('n_samples', 0),
                    'features': metrics.get('n_features', 0)
                }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return {}



