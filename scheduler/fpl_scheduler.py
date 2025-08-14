import schedule
import time
import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict
import os
from sqlalchemy import create_engine
from dotenv import load_dotenv

from data.fpl_api_client import FPLDataClient
from models.prediction.player_predictor import PlayerPredictor
from models.optimization.squad_optimizer import SquadOptimizer
from transfers.transfer_optimizer import TransferOptimizer
from database.connection import DatabaseManager

load_dotenv()

logger = logging.getLogger(__name__)

class FPLScheduler:
    """
    Automated workflow system for FPL Optimizer with intelligent scheduling
    based on FPL deadlines, data availability, and system requirements.
    """
    
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.data_collector = FPLDataClient()
        self.predictor = PlayerPredictor()
        self.optimizer = SquadOptimizer()
        self.transfer_optimizer = TransferOptimizer()
        
        # Scheduler state tracking
        self.last_data_update = None
        self.last_prediction_update = None
        self.last_model_retrain = None
        
        # Initialize logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/scheduler.log'),
                logging.StreamHandler()
            ]
        )
    
    async def daily_data_update(self):
        """Comprehensive daily data update with error handling."""
        try:
            logger.info(f"[{datetime.now()}] Starting daily data collection...")
            
            # Update FPL data
            await self.data_collector.fetch_all_data()
            
            # Process and validate new data
            validation_results = await self.data_collector.validate_data_quality()
            
            if validation_results['is_valid']:
                await self.data_collector.process_new_data()
                
                # Update player availability based on team news
                await self.update_player_availability()
                
                # Check for price changes
                price_changes = await self.data_collector.detect_price_changes()
                if price_changes:
                    logger.info(f"Detected {len(price_changes)} price changes")
                    await self.notify_price_changes(price_changes)
                
                self.last_data_update = datetime.now()
                logger.info("Daily data update completed successfully")
                
            else:
                logger.error(f"Data validation failed: {validation_results['errors']}")
                await self.notify_data_quality_issues(validation_results)
                
        except Exception as e:
            logger.error(f"Error in daily data update: {e}")
            await self.notify_system_error("daily_data_update", str(e))
    
    async def weekly_prediction_update(self):
        """Generate and validate weekly predictions."""
        try:
            logger.info(f"[{datetime.now()}] Generating weekly predictions...")
            
            current_gw = await self.get_current_gameweek()
            
            # Generate predictions for current and next gameweek
            for gw in [current_gw, current_gw + 1]:
                logger.info(f"Predicting gameweek {gw}...")
                predictions = await asyncio.to_thread(
                    self.predictor.predict_gameweek_points, gw
                )
                
                # Validate prediction quality
                quality_score = await self.validate_prediction_quality(predictions)
                
                if quality_score > 0.7:  # Quality threshold
                    await self.save_predictions(predictions, gw)
                    logger.info(f"GW{gw} predictions saved (quality: {quality_score:.2f})")
                else:
                    logger.warning(f"GW{gw} prediction quality below threshold: {quality_score:.2f}")
            
            # Update captain recommendations
            await self.update_captain_recommendations(current_gw)
            
            self.last_prediction_update = datetime.now()
            logger.info("Weekly predictions completed")
            
        except Exception as e:
            logger.error(f"Error in weekly prediction update: {e}")
            await self.notify_system_error("weekly_prediction_update", str(e))
    
    async def injury_news_update(self):
        """Check and process injury news and team updates."""
        try:
            logger.info(f"[{datetime.now()}] Checking injury news and team updates...")
            
            # Fetch latest team news
            team_news = await self.data_collector.fetch_team_news()
            
            # Process injury updates
            injury_updates = []
            availability_changes = []
            
            for news_item in team_news:
                if self.is_injury_related(news_item):
                    injury_updates.append(news_item)
                    
                    # Update player availability
                    player_id = news_item['player_id']
                    new_availability = self.calculate_availability_score(news_item)
                    
                    await self.update_player_availability_score(player_id, new_availability)
                    availability_changes.append({
                        'player_id': player_id,
                        'player_name': news_item['player_name'],
                        'old_availability': news_item.get('previous_availability', 1.0),
                        'new_availability': new_availability,
                        'reason': news_item['news_text']
                    })
            
            # Notify about significant availability changes
            if availability_changes:
                await self.notify_availability_changes(availability_changes)
            
            logger.info(f"Processed {len(injury_updates)} injury-related updates")
            
        except Exception as e:
            logger.error(f"Error in injury news update: {e}")
            await self.notify_system_error("injury_news_update", str(e))
    
    async def pre_deadline_final_update(self):
        """Comprehensive update before FPL deadline."""
        try:
            logger.info(f"[{datetime.now()}] Starting pre-deadline final update...")
            
            current_gw = await self.get_current_gameweek()
            deadline = await self.get_gameweek_deadline(current_gw)
            
            # Final data refresh
            await self.daily_data_update()
            
            # Generate last-minute predictions
            await self.weekly_prediction_update()
            
            # Check for any last-minute team news
            await self.injury_news_update()
            
            # Generate final transfer recommendations
            await self.generate_final_transfer_recommendations(current_gw)
            
            # Send deadline reminders
            await self.send_deadline_notifications(deadline)
            
            logger.info("Pre-deadline update completed")
            
        except Exception as e:
            logger.error(f"Error in pre-deadline update: {e}")
            await self.notify_system_error("pre_deadline_update", str(e))
    
    async def weekly_model_retrain(self):
        """Weekly model retraining with performance validation."""
        try:
            logger.info(f"[{datetime.now()}] Starting weekly model retraining...")
            
            # Check if enough new data is available
            data_check = await self.check_training_data_availability()
            
            if not data_check['sufficient_data']:
                logger.info(f"Insufficient new data for retraining: {data_check['reason']}")
                return
            
            # Backup current models
            await self.backup_current_models()
            
            # Retrain models
            training_results = {}
            
            for position in ['goalkeeper', 'defender', 'midfielder', 'forward']:
                logger.info(f"Retraining {position} model...")
                
                result = await asyncio.to_thread(
                    self.predictor.retrain_position_model, 
                    position
                )
                
                training_results[position] = result
                
                # Validate new model performance
                if result['performance']['cv_mae'] > result['previous_performance']['cv_mae'] * 1.1:
                    logger.warning(f"{position} model performance degraded, reverting...")
                    await self.revert_model(position)
                else:
                    logger.info(f"{position} model improved: MAE {result['performance']['cv_mae']:.3f}")
            
            self.last_model_retrain = datetime.now()
            await self.notify_model_retrain_completion(training_results)
            
            logger.info("Weekly model retraining completed")
            
        except Exception as e:
            logger.error(f"Error in model retraining: {e}")
            await self.notify_system_error("model_retrain", str(e))
    
    async def system_health_check(self):
        """Comprehensive system health monitoring."""
        try:
            logger.info(f"[{datetime.now()}] Running system health check...")
            
            health_status = {
                'database': await self.check_database_health(),
                'api_endpoints': await self.check_api_health(),
                'ml_models': await self.check_model_health(),
                'data_freshness': await self.check_data_freshness(),
                'storage_space': await self.check_storage_space(),
                'memory_usage': await self.check_memory_usage()
            }
            
            # Calculate overall health score
            health_scores = [status['score'] for status in health_status.values()]
            overall_health = sum(health_scores) / len(health_scores)
            
            if overall_health < 0.8:
                await self.notify_health_degradation(health_status, overall_health)
            
            # Log health metrics
            await self.log_health_metrics(health_status, overall_health)
            
            logger.info(f"System health check completed: {overall_health:.2f}/1.0")
            
        except Exception as e:
            logger.error(f"Error in system health check: {e}")
    
    def setup_schedule(self):
        """Configure automated scheduling based on FPL calendar."""
        
        # Daily tasks
        schedule.every(6).hours.do(asyncio.run, self.daily_data_update())
        schedule.every().day.at("02:00").do(asyncio.run, self.system_health_check())
        
        # Injury news monitoring (more frequent during season)
        schedule.every().day.at("06:00").do(asyncio.run, self.injury_news_update())
        schedule.every().day.at("14:00").do(asyncio.run, self.injury_news_update())
        schedule.every().day.at("18:00").do(asyncio.run, self.injury_news_update())
        
        # Weekly prediction updates (after matches are completed)
        schedule.every().monday.at("10:00").do(asyncio.run, self.weekly_prediction_update())
        
        # Model retraining (weekly, off-peak hours)
        schedule.every().sunday.at("03:00").do(asyncio.run, self.weekly_model_retrain())
        
        # Pre-deadline updates (Friday evening before typical deadline)
        schedule.every().friday.at("18:00").do(asyncio.run, self.pre_deadline_final_update())
        
        logger.info("Automated schedule configured successfully")
    
    # Helper methods
    
    async def get_current_gameweek(self) -> int:
        """Get current gameweek from database or API."""
        try:
            return await self.data_collector.get_current_gameweek()
        except Exception:
            return 1  # Fallback
    
    async def get_gameweek_deadline(self, gameweek: int) -> datetime:
        """Get deadline for specific gameweek."""
        query = """
        SELECT deadline_time 
        FROM gameweeks 
        WHERE id = :gameweek_id
        """
        result = await self.db_manager.fetch_one(query, {'gameweek_id': gameweek})
        return result['deadline_time'] if result else datetime.now() + timedelta(days=7)
    
    def is_injury_related(self, news_item: Dict) -> bool:
        """Check if news item is injury-related."""
        injury_keywords = ['injury', 'injured', 'doubt', 'doubtful', 'suspended', 'ban', 'strain', 'knock']
        news_text = news_item.get('news_text', '').lower()
        return any(keyword in news_text for keyword in injury_keywords)
    
    def calculate_availability_score(self, news_item: Dict) -> float:
        """Calculate player availability score based on news."""
        news_text = news_item.get('news_text', '').lower()
        
        if any(word in news_text for word in ['ruled out', 'suspended', 'banned']):
            return 0.0
        elif any(word in news_text for word in ['doubtful', 'doubt', 'fitness test']):
            return 0.5
        elif any(word in news_text for word in ['minor knock', 'slight', 'expected to play']):
            return 0.8
        else:
            return 1.0
    
    # Notification methods
    
    async def notify_price_changes(self, price_changes: List[Dict]):
        """Notify about detected price changes."""
        logger.info(f"Price changes detected: {len(price_changes)} players affected")
        # Implementation would send notifications via email, webhook, etc.
    
    async def notify_availability_changes(self, changes: List[Dict]):
        """Notify about player availability changes."""
        logger.info(f"Availability changes: {len(changes)} players affected")
        # Implementation would send notifications
    
    async def notify_system_error(self, component: str, error_message: str):
        """Notify about system errors."""
        logger.error(f"System error in {component}: {error_message}")
        # Implementation would send alerts
    
    async def send_deadline_notifications(self, deadline: datetime):
        """Send deadline reminder notifications."""
        logger.info(f"Deadline notification sent for {deadline}")
        # Implementation would send deadline reminders

def run_scheduler():
    """Main scheduler execution function."""
    scheduler = FPLScheduler()
    scheduler.setup_schedule()
    
    logger.info("FPL Optimizer Scheduler started")
    
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    except KeyboardInterrupt:
        logger.info("Scheduler stopped by user")
    except Exception as e:
        logger.error(f"Scheduler error: {e}")
        raise

if __name__ == "__main__":
    run_scheduler()
