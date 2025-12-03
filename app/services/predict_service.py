import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from app.services.model_service import load_model
from app.services.sequence_service import create_input_sequence
from app.services.polygon_service import get_last_n_minutes_5
from app.config.constants import DEVICE, PRED_INDEX, INPUT_LEN
from app.utils.date_utils import format_response

async def predict_live_price(symbol: str):
    model = load_model(symbol)

    # 1. Load last 5-min candles (OHLCV + timestamp)
    candles = get_last_n_minutes_5(symbol, INPUT_LEN)

    # extract last candle info
    last_time = candles[-1]["t"]
    last_close = candles[-1]["c"]

    start_time = candles[0]["t"]
    start_close = candles[0]["c"]


    # 2. Prepare OHLCV for the model
    prices = np.array([[c["o"], c["h"], c["l"], c["c"], c["v"]] for c in candles])

    # 3. Scale features
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(prices)

    # 4. Build input sequence
    X = create_input_sequence(scaled)
    X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)

    # 5. Predict
    with torch.no_grad():
        y_pred = model(X_tensor).cpu().numpy()[0]

    # 6. Convert predictions back to prices
    predicted_price = []
    for i in range(len(y_pred)):
        unscaled_row = np.zeros((1, 5))
        unscaled_row[0, 3] = y_pred[i]  # close index
        price = scaler.inverse_transform(unscaled_row)[0][3]
        predicted_price.append(float(price))

    # 7. Return also last timestamp + close
    return format_response(symbol, start_time, start_close, last_time, last_close, predicted_price)
