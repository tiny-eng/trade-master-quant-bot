import requests
from app.config.settings import POLYGON_API_KEY, BASE_PORIGON_URL
import datetime


def get_last_n_minutes(symbol: str, input_len: int):
    symbol = symbol.upper()
    candles = []

    days_back = 0
    while len(candles) < input_len:
        day = (datetime.datetime.now() - datetime.timedelta(days=days_back)).strftime("%Y-%m-%d")
        url = f"{BASE_PORIGON_URL}/{symbol}/range/1/minute/{day}/{day}"
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

        candles.extend(day_candles)

        days_back += 1

        if days_back > 30:
            break

    if len(candles) < input_len:
        raise Exception(f"Not enough candle data. Requested: {input_len}, Got: {len(candles)}")

    return list(reversed(candles[-input_len:]))
