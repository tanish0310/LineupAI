# config/db.py

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = "postgresql+psycopg2://fpl_user:fpl_pass@localhost:5432/fpl_db"

engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)()
Base = declarative_base()
