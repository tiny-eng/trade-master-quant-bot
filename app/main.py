from fastapi import FastAPI
from app.routers.predict_router import router as predict_router
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="AI Prediction Server", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict_router)

@app.get("/")
def home():
    return {"message": "AI server running"}