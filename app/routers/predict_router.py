from fastapi import APIRouter, HTTPException, Query
from app.services.predict_service import predict_live_price

VALID_TARGETS = [1, 5, 15, 30, 60]

router = APIRouter(prefix="/predict", tags=["Prediction"])

@router.get("/live")
async def predict_live(
    symbol: str = Query(..., description="Stock symbol"),
    target: int = Query(..., description=f"Target minute ahead. Must be one of {VALID_TARGETS}")
):
    if target not in VALID_TARGETS:
        raise HTTPException(status_code=400, detail=f"Target must be one of {VALID_TARGETS}")

    try:
        return await predict_live_price(symbol, target)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


