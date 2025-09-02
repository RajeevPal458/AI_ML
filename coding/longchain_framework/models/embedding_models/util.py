import os
from dotenv import load_dotenv, find_dotenv

try:
    # Specify the path to your .env file
    env_path = "C:/DriveD/projects/secretKeys/.env"
    
    # Load environment variables
    load_dotenv(dotenv_path=env_path)

    # Access environment variables
    my_var = os.getenv("OPENAI_API_KEY")
    print(f"OPENAI_API_KEY: {my_var}")
    
    my_var = os.getenv("ANTHROPIC_API_KEY")
    print(f"ANTHROPIC_API_KEY: {my_var}")
    
    my_var = os.getenv("AZURE_OPENAI_API_KEY")
    print(f"AZURE_OPENAI_API_KEY: {my_var}")
    
    my_var = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    print(f"HUGGINGFACEHUB_API_TOKEN: {my_var}")
    

except FileNotFoundError:
    print(f".env file not found at: {env_path}")
except Exception as e:
    print(f"An error occurred while loading .env: {e}")
