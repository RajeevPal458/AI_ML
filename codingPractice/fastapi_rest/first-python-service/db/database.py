from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
import os

url = os.getenv("DATABASE_URL", "postgresql://postgres:admin@localhost:5432/testDB")
engine = create_engine(url=url, echo=True)

LocalSession = sessionmaker(bind=engine,autocommit=False,autoflush=False)
Base = declarative_base()