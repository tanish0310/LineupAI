# models/schema.py

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# DB connection (replace with your actual credentials)
DATABASE_URL = "postgresql://postgres:king2918@localhost:5432/fpl_db"

engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
Base = declarative_base()


from sqlalchemy import Column, Integer, String, Boolean, TIMESTAMP, DECIMAL, Float
from config.db import Base

class Players(Base):
    __tablename__ = "players"
    id = Column(Integer, primary_key=True)
    name = Column(String(100))
    position = Column(Integer)  # 1=GK, 2=DEF, 3=MID, 4=FWD
    team = Column(Integer)
    price = Column(Integer)  # in tenths (e.g., 50 = £5.0m)
    total_points = Column(Integer)
    ownership_percent = Column(DECIMAL)
    form = Column(DECIMAL)
    created_at = Column(TIMESTAMP)

class Gameweeks(Base):
    __tablename__ = "gameweeks"
    id = Column(Integer, primary_key=True)
    name = Column(String(20))
    deadline_time = Column(TIMESTAMP)
    finished = Column(Boolean)
    is_current = Column(Boolean)

class GameweekStats(Base):
    __tablename__ = "gameweek_stats"
    player_id = Column(Integer, primary_key=True)
    gameweek_id = Column(Integer, primary_key=True)
    points = Column(Integer)
    minutes = Column(Integer)
    goals = Column(Integer)
    assists = Column(Integer)
    clean_sheets = Column(Integer)
    saves = Column(Integer)
    bonus = Column(Integer)
    bps = Column(Integer)

class Fixtures(Base):
    __tablename__ = "fixtures"
    id = Column(Integer, primary_key=True)
    gameweek_id = Column(Integer)
    team_h = Column(Integer)
    team_a = Column(Integer)
    team_h_difficulty = Column(Integer)
    team_a_difficulty = Column(Integer)
    kickoff_time = Column(TIMESTAMP)

class Predictions(Base):
    __tablename__ = "predictions"
    player_id = Column(Integer, primary_key=True)
    gameweek_id = Column(Integer, primary_key=True)
    predicted_points = Column(DECIMAL)
    confidence_score = Column(DECIMAL)
    created_at = Column(TIMESTAMP)

class UserSquads(Base):
    __tablename__ = "user_squads"
    user_id = Column(String(50), primary_key=True)
    gameweek_id = Column(Integer, primary_key=True)
    player_id = Column(Integer, primary_key=True)
    is_starter = Column(Boolean)
    is_captain = Column(Boolean)
    is_vice_captain = Column(Boolean)
    purchase_price = Column(Integer)
