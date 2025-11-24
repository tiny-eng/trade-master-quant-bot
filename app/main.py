from fastapi import FastAPI
from app.routers.predict_router import router as predict_router


app = FastAPI(title="AI Prediction Server", version="1.0.0")

app.include_router(predict_router)

@app.get("/")
def home():
    return {"message": "AI server running"}