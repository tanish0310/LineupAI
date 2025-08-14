from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import asyncio
import uvicorn
import logging
from typing import Dict, List, Optional, Union, Any
from contextlib import asynccontextmanager
import redis
import os
from datetime import datetime, timedelta
from prometheus_client import Counter, Histogram, generate_latest
import time
import json

# Import all our modules
from models.prediction.player_predictor import PlayerPredictor
from models.optimization.squad_optimizer import SquadOptimizer
from transfers.transfer_optimizer import TransferOptimizer
from transfers.hit_analyzer import TransferHitAnalyzer
from transfers.transfer_reasoning import TransferReasoning
from planning.multi_gw_planner import MultiGameweekPlanner
from data.fpl_api_client import FPLDataClient
from database.connection import DatabaseManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Metrics for monitoring
REQUEST_COUNT = Counter('api_requests_total', 'Total API requests', ['method', 'endpoint'])
REQUEST_LATENCY = Histogram('api_request_duration_seconds', 'Request duration')
PREDICTION_COUNT = Counter('predictions_generated_total', 'Total predictions generated')
OPTIMIZATION_COUNT = Counter('optimizations_performed_total', 'Total optimizations performed')

# Global components
predictor = None
squad_optimizer = None
transfer_optimizer = None
hit_analyzer = None
reasoning_engine = None
multi_gw_planner = None
fpl_client = None
db_manager = None
redis_client = None

# Security
security = HTTPBearer()

# Pydantic Models for enhanced type safety and API documentation
class Player(BaseModel):
    id: int
    name: str
    position: int  # 1=GK, 2=DEF, 3=MID, 4=FWD
    team: int
    price: float
    predicted_points: Optional[float] = None
    ownership: Optional[float] = None
    form: Optional[float] = None
    
class SquadPlayer(BaseModel):
    id: int
    name: str
    position: int
    team: int
    price: float
    selling_price: Optional[float] = None
    predicted_points: Optional[float] = None
    is_starter: bool = False
    is_captain: bool = False
    is_vice_captain: bool = False
    ownership: Optional[float] = None
    form: Optional[float] = None

class OptimalSquadResponse(BaseModel):
    starting_xi: Dict[str, List[SquadPlayer]]
    bench: List[SquadPlayer]
    captain: SquadPlayer
    vice_captain: SquadPlayer
    formation: str
    total_cost: float
    predicted_points: float
    budget_remaining: float
    squad_analysis: Dict[str, Any]
    validation: Dict[str, Any]
    captain_options: List[Dict[str, Any]]
    
class TransferOption(BaseModel):
    player_out: SquadPlayer
    player_in: SquadPlayer
    point_improvement: float
    cost: float
    reasoning: List[str]
    confidence: float
    
class TransferRecommendation(BaseModel):
    transfers: List[TransferOption]
    total_improvement: float
    total_cost: float
    transfers_used: int
    hit_required: bool
    hit_cost: int
    net_improvement: float
    recommendation: str  # "RECOMMENDED", "MARGINAL", "AVOID"
    
class TransferAnalysisResponse(BaseModel):
    current_squad_analysis: Dict[str, Any]
    transfer_recommendations: List[TransferRecommendation]
    hit_analysis: Optional[Dict[str, Any]] = None
    multi_gw_impact: Optional[Dict[str, Any]] = None
    injury_concerns: List[Dict[str, Any]]
    price_change_alerts: List[Dict[str, Any]]
    
class SquadAnalysisRequest(BaseModel):
    player_ids: List[int]
    gameweek_id: int
    free_transfers: int = 1
    budget_remaining: float = 0.0
    wildcard_active: bool = False
    
class PlayerAnalysisResponse(BaseModel):
    player_data: Dict[str, Any]
    prediction: Dict[str, Any]
    captain_analysis: Dict[str, Any]
    upcoming_fixtures: List[Dict[str, Any]]
    recent_performance: List[Dict[str, Any]]
    recommendation: str
    reasoning: List[str]

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle."""
    # Startup
    logger.info("Starting FPL Optimizer Pro API...")
    await initialize_components()
    logger.info("All components initialized successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down FPL Optimizer Pro API...")
    await cleanup_components()
    logger.info("Cleanup completed")

# Create FastAPI app with production settings
app = FastAPI(
    title="FPL Optimizer Pro API",
    description="Advanced Fantasy Premier League optimization system with AI-powered predictions and strategic planning",
    version="1.0.0",
    docs_url="/docs" if os.getenv("ENVIRONMENT") != "production" else None,
    redoc_url="/redoc" if os.getenv("ENVIRONMENT") != "production" else None,
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if os.getenv("ENVIRONMENT") != "production" else [
        "https://fpl-optimizer.com",
        "https://www.fpl-optimizer.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Middleware for request tracking
@app.middleware("http")
async def track_requests(request, call_next):
    start_time = time.time()
    
    # Track request
    REQUEST_COUNT.labels(
        method=request.method, 
        endpoint=request.url.path
    ).inc()
    
    response = await call_next(request)
    
    # Track latency
    process_time = time.time() - start_time
    REQUEST_LATENCY.observe(process_time)
    
    response.headers["X-Process-Time"] = str(process_time)
    return response

async def initialize_components():
    """Initialize all system components."""
    global predictor, squad_optimizer, transfer_optimizer, hit_analyzer
    global reasoning_engine, multi_gw_planner, fpl_client, db_manager, redis_client
    
    try:
        # Initialize database
        db_manager = DatabaseManager()
        await db_manager.connect()
        
        # Initialize Redis for caching
        redis_client = redis.Redis(
            host=os.getenv('REDIS_HOST', 'localhost'),
            port=int(os.getenv('REDIS_PORT', 6379)),
            decode_responses=True
        )
        
        # Initialize FPL API client
        fpl_client = FPLDataClient()
        
        # Initialize ML components
        predictor = PlayerPredictor()
        
        # Initialize optimization components
        squad_optimizer = SquadOptimizer()
        transfer_optimizer = TransferOptimizer()
        hit_analyzer = TransferHitAnalyzer()
        reasoning_engine = TransferReasoning()
        multi_gw_planner = MultiGameweekPlanner()
        
        logger.info("All components initialized successfully")
        
    except Exception as e:
        logger.error(f"Error initializing components: {e}")
        raise

async def cleanup_components():
    """Cleanup system components."""
    try:
        if db_manager:
            await db_manager.disconnect()
        if redis_client:
            redis_client.close()
        logger.info("Components cleaned up successfully")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")

async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API key for authentication."""
    if os.getenv("ENVIRONMENT") == "development":
        return True  # Skip auth in development
    
    valid_api_keys = os.getenv("API_KEYS", "").split(",")
    if credentials.credentials not in valid_api_keys:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    return True

# Helper functions
async def get_cached_predictions(gameweek_id: int) -> Dict:
    """Get predictions with Redis caching."""
    cache_key = f"predictions:{gameweek_id}"
    cached_predictions = redis_client.get(cache_key)
    
    if cached_predictions:
        try:
            return json.loads(cached_predictions)
        except json.JSONDecodeError:
            logger.warning(f"Invalid cached data for {cache_key}, regenerating...")
    
    predictions = await asyncio.to_thread(
        predictor.predict_gameweek_points,
        gameweek_id
    )
    
    # Cache for 1 hour
    redis_client.setex(cache_key, 3600, json.dumps(predictions))
    PREDICTION_COUNT.inc()
    return predictions

async def get_multi_gw_predictions(start_gw: int, horizon: int) -> Dict:
    """Get predictions for multiple gameweeks."""
    multi_predictions = {}
    for gw in range(start_gw, start_gw + horizon):
        multi_predictions[gw] = await get_cached_predictions(gw)
    return multi_predictions

def format_player(player_data, predictions: Dict) -> SquadPlayer:
    """Format player data for API response."""
    predicted_points = predictions.get(player_data.id, {}).get('points', 0.0) if predictions else 0.0
    
    return SquadPlayer(
        id=player_data.id,
        name=player_data.name,
        position=player_data.position,
        team=player_data.team,
        price=player_data.price,
        predicted_points=predicted_points,
        selling_price=getattr(player_data, 'selling_price', player_data.price),
        ownership=getattr(player_data, 'ownership', None),
        form=getattr(player_data, 'form', None)
    )

def format_starting_xi(starting_xi_data, predictions: Dict) -> Dict[str, List[SquadPlayer]]:
    """Format starting XI for response."""
    return {
        'GK': [format_player(p, predictions) for p in starting_xi_data.get('gk', [])],
        'DEF': [format_player(p, predictions) for p in starting_xi_data.get('def', [])],
        'MID': [format_player(p, predictions) for p in starting_xi_data.get('mid', [])],
        'FWD': [format_player(p, predictions) for p in starting_xi_data.get('fwd', [])]
    }

def format_bench(bench_data, predictions: Dict) -> List[SquadPlayer]:
    """Format bench players for response."""
    return [format_player(p, predictions) for p in bench_data]

# Health check endpoints
@app.get("/health")
async def health_check():
    """Comprehensive health check."""
    try:
        # Check database
        db_status = await db_manager.health_check() if db_manager else False
        
        # Check Redis
        redis_status = redis_client.ping() if redis_client else False
        
        # Check ML models
        models_status = all([
            predictor is not None,
            squad_optimizer is not None,
            transfer_optimizer is not None
        ])
        
        overall_status = all([db_status, redis_status, models_status])
        
        return {
            "status": "healthy" if overall_status else "degraded",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "database": "healthy" if db_status else "unhealthy",
                "redis": "healthy" if redis_status else "unhealthy",
                "ml_models": "healthy" if models_status else "unhealthy"
            },
            "version": "1.0.0"
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "unhealthy", "error": str(e)}
        )

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return generate_latest()

# Core prediction endpoints
@app.post("/predictions/generate")
async def generate_predictions(
    request: Dict,
    background_tasks: BackgroundTasks,
    authenticated: bool = Depends(verify_api_key)
):
    """Generate player predictions for specified gameweek."""
    try:
        gameweek_id = request.get('gameweek_id')
        if not gameweek_id:
            raise HTTPException(status_code=400, detail="gameweek_id required")
        
        # Get predictions with caching
        predictions = await get_cached_predictions(gameweek_id)
        
        # Update predictions in background
        background_tasks.add_task(update_prediction_accuracy, gameweek_id, predictions)
        
        # Format predictions with player details
        formatted_predictions = []
        for player_id, prediction in predictions.items():
            try:
                player_data = await db_manager.get_player_data(player_id)
                formatted_predictions.append({
                    'player_id': player_id,
                    'name': player_data.get('name', 'Unknown'),
                    'team': player_data.get('team', 0),
                    'position': player_data.get('position', 0),
                    'price': player_data.get('price', 0.0),
                    'predicted_points': prediction.get('points', 0.0),
                    'confidence': prediction.get('confidence', 0.0)
                })
            except Exception as e:
                logger.warning(f"Error formatting player {player_id}: {e}")
                continue
        
        return {
            "predictions": sorted(formatted_predictions, 
                                key=lambda x: x['predicted_points'], reverse=True),
            "gameweek_id": gameweek_id,
            "generated_at": datetime.now().isoformat(),
            "total_players": len(formatted_predictions)
        }
        
    except Exception as e:
        logger.error(f"Error generating predictions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/squad/optimize", response_model=OptimalSquadResponse)
async def optimize_squad(
    request: Dict,
    authenticated: bool = Depends(verify_api_key)
):
    """Enhanced squad optimization with complete FPL workflow."""
    try:
        budget = request.get('budget', 1000)  # £100m in tenths
        gameweek_id = request.get('gameweek_id')
        locked_players = request.get('locked_players', [])
        strategy = request.get('strategy', 'balanced')
        target_formation = request.get('formation', None)
        
        if not gameweek_id:
            raise HTTPException(status_code=400, detail="gameweek_id required")
        
        # Check cache first for non-custom requests
        cache_key = f"optimal_squad:{gameweek_id}:{budget}:{strategy}:{target_formation}"
        if not locked_players:
            cached_result = redis_client.get(cache_key)
            if cached_result:
                logger.info(f"Returning cached optimal squad for GW{gameweek_id}")
                try:
                    return OptimalSquadResponse(**json.loads(cached_result))
                except Exception as e:
                    logger.warning(f"Error parsing cached result: {e}")
        
        logger.info(f"Optimizing squad for GW{gameweek_id}")
        
        # Get predictions with caching
        predictions = await get_cached_predictions(gameweek_id)
        
        # Build optimal squad
        squad_solution = await asyncio.to_thread(
            squad_optimizer.build_optimal_squad,
            predictions,
            budget,
            target_formation,
            locked_players
        )
        
        # Optimize starting XI with formation selection
        starting_xi = await asyncio.to_thread(
            squad_optimizer.optimize_starting_xi,
            squad_solution['squad_15'],
            predictions
        )
        
        # Get captain recommendations
        captain_recs = await asyncio.to_thread(
            squad_optimizer.recommend_captain_vice,
            starting_xi['players'],
            predictions
        )
        
        # Generate squad analysis
        analysis = squad_optimizer.generate_squad_analysis(squad_solution)
        
        # Validate squad
        validation = squad_optimizer.validate_squad(squad_solution['squad_15'])
        
        # Format captain options
        captain_options = []
        for i, rec in enumerate(captain_recs[:3]):
            captain_options.append({
                'rank': i + 1,
                'player': format_player(rec['player'], predictions),
                'expected_points': rec.get('expected_points', 0.0),
                'safety_score': rec.get('safety_score', 0.0),
                'reasoning': rec.get('reasoning', [])
            })
        
        # Create response
        response = OptimalSquadResponse(
            starting_xi=format_starting_xi(starting_xi, predictions),
            bench=format_bench(squad_solution['bench'], predictions),
            captain=format_player(captain_recs[0]['player'], predictions),
            vice_captain=format_player(captain_recs[1]['player'], predictions),
            formation=starting_xi['formation'],
            total_cost=squad_solution['total_cost'],
            predicted_points=squad_solution['total_predicted_points'],
            budget_remaining=budget - squad_solution['total_cost'],
            squad_analysis=analysis,
            validation=validation,
            captain_options=captain_options
        )
        
        # Cache result for 30 minutes if no locked players
        if not locked_players:
            redis_client.setex(cache_key, 1800, json.dumps(response.dict()))
        
        OPTIMIZATION_COUNT.inc()
        return response
        
    except Exception as e:
        logger.error(f"Error optimizing squad: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/squad/analyze")
async def analyze_squad(request: SquadAnalysisRequest, authenticated: bool = Depends(verify_api_key)):
    """Analyze current squad performance and potential improvements."""
    try:
        # Get predictions
        predictions = await get_cached_predictions(request.gameweek_id)
        
        # Convert player IDs to squad objects
        current_squad = []
        for player_id in request.player_ids:
            try:
                player_data = await db_manager.get_player_data(player_id)
                current_squad.append(player_data)
            except Exception as e:
                logger.warning(f"Could not load player {player_id}: {e}")
                continue
        
        # Analyze current squad
        analysis = await asyncio.to_thread(
            transfer_optimizer.analyze_current_squad,
            current_squad, predictions
        )
        
        # Check for injury concerns
        injury_concerns = await check_injury_concerns(current_squad)
        
        # Check for price change alerts
        price_alerts = await check_price_changes(current_squad)
        
        return {
            "current_points": analysis['current_points'],
            "optimal_points": analysis['optimal_points'],
            "improvement_potential": analysis['improvement_potential'],
            "squad_rank_percentile": analysis['squad_rank_percentile'],
            "injury_concerns": injury_concerns,
            "price_change_alerts": price_alerts,
            "analysis_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Squad analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/transfers/analyze", response_model=TransferAnalysisResponse)
async def analyze_transfers(
    request: Dict,
    authenticated: bool = Depends(verify_api_key)
):
    """Enhanced transfer analysis with detailed reasoning."""
    try:
        current_squad = request.get('current_squad')
        gameweek_id = request.get('gameweek_id')
        free_transfers = request.get('free_transfers', 1)
        budget_remaining = request.get('budget_remaining', 0)
        analyze_hits = request.get('analyze_hits', True)
        multi_gw_horizon = request.get('multi_gw_horizon', 3)
        
        if not all([current_squad, gameweek_id]):
            raise HTTPException(status_code=400, detail="current_squad and gameweek_id required")
        
        logger.info(f"Enhanced transfer analysis for GW{gameweek_id}")
        
        # Get predictions for current gameweek
        predictions = await get_cached_predictions(gameweek_id)
        
        # Analyze current squad
        squad_analysis = await asyncio.to_thread(
            transfer_optimizer.analyze_current_squad,
            current_squad,
            predictions
        )
        
        # Generate transfer recommendations
        recommendations = await asyncio.to_thread(
            transfer_optimizer.suggest_transfers,
            current_squad,
            free_transfers,
            predictions,
            budget_remaining
        )
        
        # Enhanced recommendations with reasoning
        enhanced_recommendations = []
        for rec in recommendations[:5]:  # Top 5 recommendations
            if rec.get('players_out') and rec.get('players_in'):
                transfers = []
                
                for player_out, player_in in zip(rec['players_out'], rec['players_in']):
                    reasoning = await asyncio.to_thread(
                        reasoning_engine.generate_transfer_explanation,
                        player_out,
                        player_in,
                        predictions,
                        gameweek_id=gameweek_id
                    )
                    
                    transfer_option = TransferOption(
                        player_out=format_player(player_out, predictions),
                        player_in=format_player(player_in, predictions),
                        point_improvement=predictions.get(player_in.id, {}).get('points', 0) - 
                                        predictions.get(player_out.id, {}).get('points', 0),
                        cost=player_in.price - player_out.selling_price,
                        reasoning=reasoning,
                        confidence=rec.get('confidence', 0.8)
                    )
                    transfers.append(transfer_option)
                
                # Calculate hit analysis
                hits_required = max(0, rec['transfers_used'] - free_transfers)
                hit_cost = hits_required * 4
                net_improvement = rec['improvement'] - hit_cost
                
                # Determine recommendation
                if net_improvement > 2:
                    recommendation = "RECOMMENDED"
                elif net_improvement > 0:
                    recommendation = "MARGINAL"
                else:
                    recommendation = "AVOID"
                
                enhanced_rec = TransferRecommendation(
                    transfers=transfers,
                    total_improvement=rec['improvement'],
                    total_cost=rec['cost'],
                    transfers_used=rec['transfers_used'],
                    hit_required=hits_required > 0,
                    hit_cost=hit_cost,
                    net_improvement=net_improvement,
                    recommendation=recommendation
                )
                enhanced_recommendations.append(enhanced_rec)
        
        # Analyze transfer hits if requested
        hit_analysis = None
        if analyze_hits and recommendations:
            hit_analysis = await asyncio.to_thread(
                hit_analyzer.analyze_hit_value,
                recommendations[0],
                await get_multi_gw_predictions(gameweek_id, multi_gw_horizon)
            )
        
        # Multi-gameweek impact analysis
        multi_gw_impact = None
        if multi_gw_horizon > 1 and enhanced_recommendations:
            multi_gw_impact = await asyncio.to_thread(
                multi_gw_planner.analyze_transfer_impact,
                enhanced_recommendations[0],
                gameweek_id,
                multi_gw_horizon
            )
        
        # Check for additional concerns
        injury_concerns = await check_injury_concerns(current_squad)
        price_alerts = await check_price_changes(current_squad)
        
        return TransferAnalysisResponse(
            current_squad_analysis=squad_analysis,
            transfer_recommendations=enhanced_recommendations,
            hit_analysis=hit_analysis,
            multi_gw_impact=multi_gw_impact,
            injury_concerns=injury_concerns,
            price_change_alerts=price_alerts
        )
        
    except Exception as e:
        logger.error(f"Error in enhanced transfer analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/transfers/hit-analysis")
async def analyze_transfer_hits(
    request: Dict,
    authenticated: bool = Depends(verify_api_key)
):
    """Comprehensive analysis of transfer hits across multiple gameweeks."""
    try:
        transfer_options = request.get('transfer_options', [])
        gameweek_id = request.get('gameweek_id')
        analysis_horizon = request.get('analysis_horizon', 6)
        
        if not all([transfer_options, gameweek_id]):
            raise HTTPException(status_code=400, detail="transfer_options and gameweek_id required")
        
        logger.info(f"Analyzing transfer hits for GW{gameweek_id}")
        
        # Generate multi-gameweek predictions
        predictions_multi_gw = await get_multi_gw_predictions(gameweek_id, analysis_horizon)
        
        # Analyze each transfer option
        hit_analyses = []
        for option in transfer_options:
            analysis = await asyncio.to_thread(
                hit_analyzer.analyze_hit_value,
                option,
                predictions_multi_gw
            )
            hit_analyses.append(analysis)
        
        # Compare scenarios
        comparison = await asyncio.to_thread(
            hit_analyzer.compare_hit_scenarios,
            transfer_options,
            predictions_multi_gw
        )
        
        return {
            "hit_analyses": hit_analyses,
            "scenario_comparison": comparison,
            "analysis_parameters": {
                "gameweek_start": gameweek_id,
                "analysis_horizon": analysis_horizon,
                "scenarios_analyzed": len(transfer_options),
                "analysis_timestamp": datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Error analyzing transfer hits: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/planning/multi-gameweek")
async def create_multi_gameweek_plan(
    request: Dict,
    authenticated: bool = Depends(verify_api_key)
):
    """Create strategic multi-gameweek plan."""
    try:
        current_squad = request.get('current_squad')
        current_gw = request.get('current_gameweek')
        planning_horizon = request.get('planning_horizon', 6)
        available_chips = request.get('available_chips', [])
        strategy_type = request.get('strategy_type', 'balanced')
        
        if not all([current_squad, current_gw]):
            raise HTTPException(status_code=400, detail="current_squad and current_gameweek required")
        
        logger.info(f"Creating {planning_horizon}-gameweek strategic plan")
        
        # Create comprehensive plan
        strategic_plan = await asyncio.to_thread(
            multi_gw_planner.create_multi_gameweek_plan,
            current_squad,
            current_gw,
            planning_horizon,
            available_chips,
            strategy_type
        )
        
        return {
            "strategic_plan": strategic_plan,
            "plan_summary": {
                "planning_horizon": planning_horizon,
                "strategy_type": strategy_type,
                "chips_available": len(available_chips),
                "total_transfers_planned": strategic_plan.get('total_transfers', 0),
                "expected_point_gain": strategic_plan.get('expected_gain', 0),
                "created_at": datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Error creating multi-gameweek plan: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/player/{player_id}/analysis", response_model=PlayerAnalysisResponse)
async def get_player_analysis(
    player_id: int,
    gameweek_id: int,
    authenticated: bool = Depends(verify_api_key)
):
    """Get comprehensive analysis for a specific player."""
    try:
        # Get player predictions
        predictions = await get_cached_predictions(gameweek_id)
        
        if player_id not in predictions:
            raise HTTPException(status_code=404, detail="Player not found")
        
        player_prediction = predictions[player_id]
        
        # Get player data from database
        player_data = await db_manager.get_player_data(player_id)
        
        # Get upcoming fixtures
        fixtures = await db_manager.get_player_upcoming_fixtures(player_id, 5)
        
        # Get recent performance
        recent_performance = await db_manager.get_player_recent_performance(player_id, 5)
        
        # Calculate captain potential
        captain_analysis = await asyncio.to_thread(
            predictor.calculate_captain_multiplier,
            player_id,
            player_prediction['points'],
            gameweek_id
        )
        
        # Generate recommendation
        recommendation, reasoning = await generate_player_recommendation(
            player_data, player_prediction, fixtures, recent_performance
        )
        
        return PlayerAnalysisResponse(
            player_data=player_data,
            prediction=player_prediction,
            captain_analysis=captain_analysis,
            upcoming_fixtures=fixtures,
            recent_performance=recent_performance,
            recommendation=recommendation,
            reasoning=reasoning
        )
        
    except Exception as e:
        logger.error(f"Error analyzing player {player_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/gameweek/current")
async def get_current_gameweek():
    """Get current gameweek information."""
    try:
        current_gw = await asyncio.to_thread(fpl_client.get_current_gameweek)
        return {
            "current_gameweek": current_gw,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting current gameweek: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/teams")
async def get_teams():
    """Get all Premier League teams."""
    try:
        teams = await asyncio.to_thread(fpl_client.get_all_teams)
        return {
            "teams": teams,
            "total_teams": len(teams),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting teams: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/data/update")
async def trigger_data_update(
    background_tasks: BackgroundTasks,
    authenticated: bool = Depends(verify_api_key)
):
    """Trigger background data update."""
    try:
        # Add background task for data update
        background_tasks.add_task(update_all_data)
        
        return {
            "status": "Data update initiated",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error triggering data update: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/performance")
async def get_model_performance():
    """Get ML model performance metrics."""
    try:
        performance_summary = predictor.get_model_performance_summary()
        
        return {
            "model_performance": performance_summary,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting model performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Helper functions for enhanced functionality
async def check_injury_concerns(squad):
    """Check for injury concerns in squad."""
    concerns = []
    for player in squad:
        try:
            injury_status = await db_manager.get_player_injury_status(player.id)
            if injury_status.get('is_injured') or injury_status.get('doubt_level', 0) > 0.3:
                concerns.append({
                    'player_id': player.id,
                    'player_name': player.name,
                    'injury_type': injury_status.get('injury_type', 'Unknown'),
                    'severity': injury_status.get('doubt_level', 0),
                    'expected_return': injury_status.get('expected_return', 'Unknown')
                })
        except Exception as e:
            logger.warning(f"Could not check injury status for player {player.id}: {e}")
    return concerns

async def check_price_changes(squad):
    """Check for potential price changes."""
    alerts = []
    for player in squad:
        try:
            price_data = await db_manager.get_price_change_data(player.id)
            if abs(price_data.get('change_probability', 0)) > 0.7:
                alerts.append({
                    'player_id': player.id,
                    'player_name': player.name,
                    'direction': 'rise' if price_data['change_probability'] > 0 else 'fall',
                    'probability': abs(price_data['change_probability']),
                    'target_price': price_data.get('target_price', player.price)
                })
        except Exception as e:
            logger.warning(f"Could not check price changes for player {player.id}: {e}")
    return alerts

async def generate_player_recommendation(player_data, prediction, fixtures, recent_performance):
    """Generate player recommendation based on analysis."""
    form_score = sum([gw.get('points', 0) for gw in recent_performance]) / max(len(recent_performance), 1)
    avg_fixture_difficulty = sum([f.get('difficulty', 3) for f in fixtures[:3]]) / max(len(fixtures[:3]), 1)
    predicted_points = prediction.get('points', 0)
    
    reasoning = []
    
    if predicted_points > 7:
        recommendation = "STRONG BUY"
        reasoning.append(f"High predicted points ({predicted_points:.1f})")
    elif predicted_points > 5:
        recommendation = "BUY"
        reasoning.append(f"Good predicted points ({predicted_points:.1f})")
    elif predicted_points < 3:
        recommendation = "SELL"
        reasoning.append(f"Low predicted points ({predicted_points:.1f})")
    else:
        recommendation = "HOLD"
        reasoning.append(f"Average predicted points ({predicted_points:.1f})")
    
    if form_score > 6:
        reasoning.append(f"Excellent recent form ({form_score:.1f} avg)")
    elif form_score < 3:
        reasoning.append(f"Poor recent form ({form_score:.1f} avg)")
    
    if avg_fixture_difficulty <= 2:
        reasoning.append("Favorable upcoming fixtures")
    elif avg_fixture_difficulty >= 4:
        reasoning.append("Difficult upcoming fixtures")
    
    return recommendation, reasoning

# Background tasks
async def update_all_data():
    """Background task to update all FPL data."""
    try:
        logger.info("Starting background data update")
        
        # Update FPL data
        await asyncio.to_thread(fpl_client.update_all_data)
        
        # Clear prediction cache
        cache_keys = redis_client.keys("predictions:*")
        if cache_keys:
            redis_client.delete(*cache_keys)
        
        # Clear squad optimization cache
        cache_keys = redis_client.keys("optimal_squad:*")
        if cache_keys:
            redis_client.delete(*cache_keys)
        
        logger.info("Background data update completed")
        
    except Exception as e:
        logger.error(f"Error in background data update: {e}")

async def update_prediction_accuracy(gameweek_id: int, predictions: Dict):
    """Background task to update prediction accuracy metrics."""
    try:
        # This would compare predictions with actual results once gameweek is finished
        logger.info(f"Updated prediction tracking for GW{gameweek_id}")
    except Exception as e:
        logger.error(f"Error updating prediction accuracy: {e}")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        workers=int(os.getenv("WORKERS", 1)),
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
        reload=os.getenv("ENVIRONMENT") == "development"
    )
