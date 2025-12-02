from fastapi import FastAPI
from app.routers.predict_router import router as predict_router
from fastapi.middleware.cors import CORSMiddleware
from app.config.settings import CORS_ORIGINS
import os

app = FastAPI(title="AI Prediction Server", version="1.0.0")

cors_env = CORS_ORIGINS
origins = [origin.strip() for origin in cors_env.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict_router)

@app.get("/")
def home():
    return {"message": "AI server running"}