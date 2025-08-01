from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime, Decimal, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import logging
from config.settings import settings

Base = declarative_base()

class Player(Base):
    __tablename__ = 'players'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    position = Column(Integer, nullable=False)  # 1=GK, 2=DEF, 3=MID, 4=FWD
    team = Column(Integer, nullable=False)
    price = Column(Integer, nullable=False)  # in tenths (e.g., 50 = £5.0m)
    total_points = Column(Integer, default=0)
    ownership_percent = Column(Decimal(5,2), default=0.0)
    form = Column(Decimal(4,2), default=0.0)
    status = Column(String(1), default='a')  # a=available, d=doubtful, i=injured, u=unavailable
    news = Column(Text, default='')
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    gameweek_stats = relationship("GameweekStats", back_populates="player")
    predictions = relationship("Prediction", back_populates="player")

class Team(Base):
    __tablename__ = 'teams'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    short_name = Column(String(3), nullable=False)
    strength = Column(Integer, default=3)
    strength_overall_home = Column(Integer, default=3)
    strength_overall_away = Column(Integer, default=3)
    strength_attack_home = Column(Integer, default=3)
    strength_attack_away = Column(Integer, default=3)
    strength_defence_home = Column(Integer, default=3)
    strength_defence_away = Column(Integer, default=3)

class Gameweek(Base):
    __tablename__ = 'gameweeks'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(20), nullable=False)
    deadline_time = Column(DateTime, nullable=False)
    finished = Column(Boolean, default=False)
    is_current = Column(Boolean, default=False)
    is_next = Column(Boolean, default=False)
    
    # Relationships
    fixtures = relationship("Fixture", back_populates="gameweek")
    gameweek_stats = relationship("GameweekStats", back_populates="gameweek")

class GameweekStats(Base):
    __tablename__ = 'gameweek_stats'
    
    player_id = Column(Integer, ForeignKey('players.id'), primary_key=True)
    gameweek_id = Column(Integer, ForeignKey('gameweeks.id'), primary_key=True)
    points = Column(Integer, default=0)
    minutes = Column(Integer, default=0)
    goals = Column(Integer, default=0)
    assists = Column(Integer, default=0)
    clean_sheets = Column(Integer, default=0)
    saves = Column(Integer, default=0)
    bonus = Column(Integer, default=0)
    bps = Column(Integer, default=0)  # Bonus Point System
    yellow_cards = Column(Integer, default=0)
    red_cards = Column(Integer, default=0)
    own_goals = Column(Integer, default=0)
    penalties_saved = Column(Integer, default=0)
    penalties_missed = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    player = relationship("Player", back_populates="gameweek_stats")
    gameweek = relationship("Gameweek", back_populates="gameweek_stats")

class Fixture(Base):
    __tablename__ = 'fixtures'
    
    id = Column(Integer, primary_key=True)
    gameweek_id = Column(Integer, ForeignKey('gameweeks.id'))
    team_h = Column(Integer, nullable=False)
    team_a = Column(Integer, nullable=False)
    team_h_difficulty = Column(Integer, nullable=False)
    team_a_difficulty = Column(Integer, nullable=False)
    kickoff_time = Column(DateTime)
    finished = Column(Boolean, default=False)
    team_h_score = Column(Integer)
    team_a_score = Column(Integer)
    
    # Relationships
    gameweek = relationship("Gameweek", back_populates="fixtures")

class Prediction(Base):
    __tablename__ = 'predictions'
    
    player_id = Column(Integer, ForeignKey('players.id'), primary_key=True)
    gameweek_id = Column(Integer, ForeignKey('gameweeks.id'), primary_key=True)
    predicted_points = Column(Decimal(5,2), nullable=False)
    confidence_score = Column(Decimal(4,3), nullable=False)
    model_version = Column(String(50), default='v1.0')
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    player = relationship("Player", back_populates="predictions")

class DatabaseManager:
    def __init__(self, database_url=None):
        self.database_url = database_url or settings.DATABASE_URL
        self.engine = create_engine(self.database_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self.logger = logging.getLogger(__name__)
    
    def create_tables(self):
        """Create all database tables"""
        try:
            Base.metadata.create_all(bind=self.engine)
            self.logger.info("Database tables created successfully")
        except Exception as e:
            self.logger.error(f"Error creating database tables: {e}")
            raise
    
    def get_session(self):
        """Get database session"""
        return self.SessionLocal()
    
    def close_session(self, session):
        """Close database session"""
        session.close()

# Initialize database manager
db_manager = DatabaseManager()
