import requests
from app.config.settings import POLYGON_API_KEY
import datetime

BASE_URL = "https://api.polygon.io/v2/aggs/ticker"

def get_last_n_minutes(symbol: str, input_len: int):
    """
    Fetch the latest `input_len` 1-minute candles for a symbol, across multiple days if needed.
    """
    symbol = symbol.upper()
    candles = []

    days_back = 0
    while len(candles) < input_len:
        day = (datetime.datetime.now() - datetime.timedelta(days=days_back)).strftime("%Y-%m-%d")
        url = f"{BASE_URL}/{symbol}/range/1/minute/{day}/{day}"
        params = {
            "adjusted": "true",
            "sort": "desc",  # newest first
            "limit": 5000,   # max per request
            "apiKey": POLYGON_API_KEY
        }

        r = requests.get(url, params=params)
        if r.status_code != 200:
            raise Exception(f"Polygon API error: {r.status_code} - {r.text}")

        data = r.json()
        day_candles = [c["c"] for c in data.get("results", [])]

        # Append newest first, reverse later
        candles.extend(day_candles)

        days_back += 1

        # Stop after too many days to prevent infinite loop
        if days_back > 30:
            break

    if len(candles) < input_len:
        raise Exception(f"Not enough candle data. Requested: {input_len}, Got: {len(candles)}")

    # Return oldest -> newest
    return list(reversed(candles[-input_len:]))
