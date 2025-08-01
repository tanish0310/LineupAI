from config.db import engine
from models.schema import Base

# This creates all tables defined using SQLAlchemy's Base
Base.metadata.create_all(engine)

print("✅ All tables created successfully!")
