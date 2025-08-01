-- Drop existing tables if they exist
DROP TABLE IF EXISTS user_squads CASCADE;
DROP TABLE IF EXISTS predictions CASCADE;
DROP TABLE IF EXISTS gameweek_stats CASCADE;
DROP TABLE IF EXISTS team_news CASCADE;
DROP TABLE IF EXISTS fixtures CASCADE;
DROP TABLE IF EXISTS players CASCADE;
DROP TABLE IF EXISTS teams CASCADE;
DROP TABLE IF EXISTS gameweeks CASCADE;
DROP TABLE IF EXISTS positions CASCADE;

-- Core reference tables
CREATE TABLE positions (
    id INTEGER PRIMARY KEY,
    singular_name VARCHAR(20) NOT NULL,
    singular_name_short VARCHAR(5) NOT NULL,
    plural_name VARCHAR(20) NOT NULL,
    plural_name_short VARCHAR(5) NOT NULL
);

CREATE TABLE teams (
    id INTEGER PRIMARY KEY,
    name VARCHAR(50) NOT NULL,
    short_name VARCHAR(10) NOT NULL,
    strength INTEGER,
    strength_overall_home INTEGER,
    strength_overall_away INTEGER,
    strength_attack_home INTEGER,
    strength_attack_away INTEGER,
    strength_defence_home INTEGER,
    strength_defence_away INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE gameweeks (
    id INTEGER PRIMARY KEY,
    name VARCHAR(20) NOT NULL,
    deadline_time TIMESTAMP,
    average_entry_score INTEGER,
    finished BOOLEAN DEFAULT FALSE,
    data_checked BOOLEAN DEFAULT FALSE,
    highest_scoring_entry INTEGER,
    deadline_day INTEGER,
    is_previous BOOLEAN DEFAULT FALSE,
    is_current BOOLEAN DEFAULT FALSE,
    is_next BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Main players table with comprehensive stats
CREATE TABLE players (
    id INTEGER PRIMARY KEY,
    first_name VARCHAR(50),
    second_name VARCHAR(50),
    web_name VARCHAR(50),
    team INTEGER REFERENCES teams(id),
    position INTEGER REFERENCES positions(id),
    now_cost INTEGER NOT NULL, -- Price in tenths (50 = £5.0m)
    cost_change_event INTEGER DEFAULT 0,
    cost_change_event_fall INTEGER DEFAULT 0,
    cost_change_start INTEGER DEFAULT 0,
    cost_change_start_fall INTEGER DEFAULT 0,
    
    -- Performance stats
    total_points INTEGER DEFAULT 0,
    event_points INTEGER DEFAULT 0,
    points_per_game DECIMAL(4,2) DEFAULT 0,
    ep_this DECIMAL(4,2) DEFAULT 0,
    ep_next DECIMAL(4,2) DEFAULT 0,
    
    -- Form and trends
    form DECIMAL(4,2) DEFAULT 0,
    form_rank INTEGER,
    form_rank_type INTEGER,
    
    -- Ownership and selection
    selected_by_percent DECIMAL(5,2) DEFAULT 0,
    transfers_in INTEGER DEFAULT 0,
    transfers_in_event INTEGER DEFAULT 0,
    transfers_out INTEGER DEFAULT 0,
    transfers_out_event INTEGER DEFAULT 0,
    
    -- Performance metrics
    goals_scored INTEGER DEFAULT 0,
    assists INTEGER DEFAULT 0,
    clean_sheets INTEGER DEFAULT 0,
    goals_conceded INTEGER DEFAULT 0,
    own_goals INTEGER DEFAULT 0,
    penalties_saved INTEGER DEFAULT 0,
    penalties_missed INTEGER DEFAULT 0,
    yellow_cards INTEGER DEFAULT 0,
    red_cards INTEGER DEFAULT 0,
    saves INTEGER DEFAULT 0,
    bonus INTEGER DEFAULT 0,
    bps INTEGER DEFAULT 0,
    
    -- Advanced metrics
    influence DECIMAL(6,2) DEFAULT 0,
    creativity DECIMAL(6,2) DEFAULT 0,
    threat DECIMAL(6,2) DEFAULT 0,
    ict_index DECIMAL(6,2) DEFAULT 0,
    
    -- Availability
    chance_of_playing_this_round INTEGER,
    chance_of_playing_next_round INTEGER,
    in_dreamteam BOOLEAN DEFAULT FALSE,
    dreamteam_count INTEGER DEFAULT 0,
    
    -- Status and news
    status VARCHAR(1) DEFAULT 'a', -- a=available, d=doubtful, i=injured, u=unavailable, s=suspended
    news TEXT DEFAULT '',
    news_added TIMESTAMP,
    
    -- Calculated fields (will be updated by data processor)
    points_per_million DECIMAL(6,2) DEFAULT 0,
    availability_score DECIMAL(4,2) DEFAULT 1.0,
    value_score DECIMAL(4,2) DEFAULT 1.0,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Fixtures with enhanced difficulty and strength metrics
CREATE TABLE fixtures (
    id INTEGER PRIMARY KEY,
    gameweek_id INTEGER REFERENCES gameweeks(id),
    kickoff_time TIMESTAMP,
    team_h INTEGER REFERENCES teams(id),
    team_a INTEGER REFERENCES teams(id),
    team_h_score INTEGER,
    team_a_score INTEGER,
    finished BOOLEAN DEFAULT FALSE,
    finished_provisional BOOLEAN DEFAULT FALSE,
    minutes INTEGER DEFAULT 0,
    provisional_start_time BOOLEAN DEFAULT FALSE,
    started BOOLEAN DEFAULT FALSE,
    
    -- Difficulty ratings (1=easiest, 5=hardest)
    team_h_difficulty INTEGER,
    team_a_difficulty INTEGER,
    
    -- Enhanced strength metrics
    home_advantage DECIMAL(3,2) DEFAULT 1.1,
    home_team_strength DECIMAL(4,2),
    away_team_strength DECIMAL(4,2),
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Detailed gameweek statistics for each player
CREATE TABLE gameweek_stats (
    player_id INTEGER REFERENCES players(id),
    gameweek_id INTEGER REFERENCES gameweeks(id),
    fixture_id INTEGER REFERENCES fixtures(id),
    
    -- Basic performance
    total_points INTEGER DEFAULT 0,
    was_home BOOLEAN,
    kickoff_time TIMESTAMP,
    team_h_score INTEGER,
    team_a_score INTEGER,
    round INTEGER,
    
    -- Detailed stats
    minutes INTEGER DEFAULT 0,
    goals_scored INTEGER DEFAULT 0,
    assists INTEGER DEFAULT 0,
    clean_sheets INTEGER DEFAULT 0,
    goals_conceded INTEGER DEFAULT 0,
    own_goals INTEGER DEFAULT 0,
    penalties_saved INTEGER DEFAULT 0,
    penalties_missed INTEGER DEFAULT 0,
    yellow_cards INTEGER DEFAULT 0,
    red_cards INTEGER DEFAULT 0,
    saves INTEGER DEFAULT 0,
    bonus INTEGER DEFAULT 0,
    bps INTEGER DEFAULT 0,
    
    -- Advanced metrics
    influence DECIMAL(6,2) DEFAULT 0,
    creativity DECIMAL(6,2) DEFAULT 0,
    threat DECIMAL(6,2) DEFAULT 0,
    ict_index DECIMAL(6,2) DEFAULT 0,
    
    -- Expected stats (if available)
    expected_goals DECIMAL(4,2),
    expected_assists DECIMAL(4,2),
    expected_goal_involvements DECIMAL(4,2),
    expected_goals_conceded DECIMAL(4,2),
    
    -- Calculated rolling metrics (updated by processor)
    points_3gw_avg DECIMAL(4,2),
    points_5gw_avg DECIMAL(4,2),
    minutes_3gw_avg DECIMAL(4,2),
    weighted_form_3gw DECIMAL(4,2),
    points_std_5gw DECIMAL(4,2),
    consistency_score DECIMAL(4,2),
    
    PRIMARY KEY (player_id, gameweek_id)
);

-- ML predictions for each player/gameweek
CREATE TABLE predictions (
    player_id INTEGER REFERENCES players(id),
    gameweek_id INTEGER REFERENCES gameweeks(id),
    predicted_points DECIMAL(5,2) NOT NULL,
    confidence_score DECIMAL(4,2) NOT NULL,
    prediction_method VARCHAR(50), -- e.g., 'xgboost_midfielder', 'ensemble'
    
    -- Position-specific predictions
    predicted_minutes DECIMAL(4,1),
    predicted_goals DECIMAL(4,2),
    predicted_assists DECIMAL(4,2),
    predicted_clean_sheets DECIMAL(4,2),
    predicted_saves DECIMAL(4,2),
    predicted_bonus DECIMAL(4,2),
    
    -- Risk factors
    injury_risk DECIMAL(4,2) DEFAULT 0,
    rotation_risk DECIMAL(4,2) DEFAULT 0,
    fixture_difficulty_impact DECIMAL(4,2) DEFAULT 1,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (player_id, gameweek_id, prediction_method)
);

-- Team news and injury updates
CREATE TABLE team_news (
    id SERIAL PRIMARY KEY,
    player_id INTEGER REFERENCES players(id),
    news_text TEXT,
    injury_status VARCHAR(20) DEFAULT 'available', -- available, doubtful, injured, suspended
    severity VARCHAR(20), -- minor, major, long_term
    expected_return_gameweek INTEGER,
    source VARCHAR(100),
    confidence_score DECIMAL(3,2) DEFAULT 0.5,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- User squad tracking
CREATE TABLE user_squads (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(50) NOT NULL,
    gameweek_id INTEGER REFERENCES gameweeks(id),
    player_id INTEGER REFERENCES players(id),
    position_in_squad INTEGER, -- 1-15 squad position
    is_starter BOOLEAN DEFAULT FALSE,
    is_captain BOOLEAN DEFAULT FALSE,
    is_vice_captain BOOLEAN DEFAULT FALSE,
    purchase_price INTEGER, -- Price when purchased
    selling_price INTEGER, -- Current selling price
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(user_id, gameweek_id, player_id)
);

-- Transfer recommendations and history
CREATE TABLE transfer_recommendations (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(50) NOT NULL,
    gameweek_id INTEGER REFERENCES gameweeks(id),
    player_out_id INTEGER REFERENCES players(id),
    player_in_id INTEGER REFERENCES players(id),
    expected_points_improvement DECIMAL(5,2),
    cost_change DECIMAL(6,2),
    hit_required BOOLEAN DEFAULT FALSE,
    hit_cost INTEGER DEFAULT 0,
    confidence_score DECIMAL(4,2),
    reasoning TEXT,
    recommendation_rank INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX idx_players_team_position ON players(team, position);
CREATE INDEX idx_players_now_cost ON players(now_cost);
CREATE INDEX idx_players_total_points ON players(total_points);
CREATE INDEX idx_gameweek_stats_player_gw ON gameweek_stats(player_id, gameweek_id);
CREATE INDEX idx_predictions_player_gw ON predictions(player_id, gameweek_id);
CREATE INDEX idx_fixtures_gameweek ON fixtures(gameweek_id);
CREATE INDEX idx_fixtures_teams ON fixtures(team_h, team_a);
CREATE INDEX idx_user_squads_user_gw ON user_squads(user_id, gameweek_id);

-- Views for common queries
CREATE VIEW current_gameweek AS
SELECT * FROM gameweeks WHERE is_current = TRUE;

CREATE VIEW player_current_form AS
SELECT 
    p.id,
    p.web_name,
    p.position,
    p.team,
    p.now_cost,
    p.total_points,
    p.form,
    p.points_per_game,
    p.selected_by_percent,
    p.points_per_million,
    t.name as team_name,
    pos.singular_name_short as position_name
FROM players p
JOIN teams t ON p.team = t.id
JOIN positions pos ON p.position = pos.id
WHERE p.status = 'a'
ORDER BY p.total_points DESC;
