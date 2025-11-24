import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from app.services.model_service import load_model
from app.services.sequence_service import create_input_sequence
from app.services.polygon_service import get_last_n_minutes
from app.config.constants import DEVICE, PRED_INDEX, INPUT_LEN


async def predict_live_price(symbol: str, target: int):
    if target not in PRED_INDEX:
        raise ValueError(f"Target must be one of: {PRED_INDEX}")

    # 1) Load model
    model = load_model(symbol)

    # 2) Get last price data
    prices = get_last_n_minutes(symbol, INPUT_LEN)
    prices = np.array(prices).reshape(-1, 1)

    # 3) Fit scaler dynamically on last prices
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(prices)  # ✅ Fit before transform

    # 4) Build input sequence
    X = create_input_sequence(scaled)
    X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)

    # 5) Predict
    with torch.no_grad():
        y_pred = model(X_tensor).cpu().numpy()[0]

    # 6) Extract specific prediction
    idx = PRED_INDEX.index(target)
    predicted_scaled = y_pred[idx].reshape(-1, 1)

    # 7) Inverse transform to original scale
    predicted_price = scaler.inverse_transform(predicted_scaled)[0][0]

    return {
        "symbol": symbol,
        "target_min": target,
        "prediction": round(float(predicted_price), 2)
    }
