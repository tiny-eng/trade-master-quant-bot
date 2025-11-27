from fastapi import APIRouter, HTTPException, Query
from app.services.predict_service import predict_live_price


router = APIRouter(prefix="/predict", tags=["Prediction"])

@router.get("/live")
async def predict_live(
    symbol: str = Query(..., description="Stock symbol"),
):
    return await predict_live_price(symbol)


