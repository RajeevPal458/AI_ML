import os
from dotenv import load_dotenv, find_dotenv

try:
    # Specify the path to your .env file
    env_path = "C:/DriveD/projects/secretKeys/.env"
    
    # Load environment variables
    load_dotenv(dotenv_path=env_path)

    # Access environment variables
    my_var = os.getenv("OPENAI_API_KEY")
    print(f"MY_VARIABLE: {my_var}")

except FileNotFoundError:
    print(f".env file not found at: {env_path}")
except Exception as e:
    print(f"An error occurred while loading .env: {e}")
