import os
from dotenv import load_dotenv

load_dotenv()  # loads variables from .env automatically
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "")
BASE_PORIGON_URL = os.getenv("BASE_URL")
