# config/settings.py

from pydantic import BaseSettings
from dotenv import load_dotenv
import os

# Load the correct .env file
env = os.getenv("ENV", "dev")
load_dotenv(dotenv_path=f".env.{env}")

class Settings(BaseSettings):
    env: str = os.getenv("ENV")
    debug: bool = os.getenv("DEBUG") == "True"
    database_url: str = os.getenv("DATABASE_URL")

settings = Settings()
