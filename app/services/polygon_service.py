import requests
from app.config.settings import POLYGON_API_KEY
import datetime

from zoneinfo import ZoneInfo

BASE_URL = "https://api.polygon.io/v2/aggs/ticker"

def get_last_n_minutes_5(symbol: str, input_len: int):
    symbol = symbol.upper()
    candles = []
    days_back = 0
    while len(candles) < input_len:
        day = (datetime.datetime.now() - datetime.timedelta(days=days_back)).strftime("%Y-%m-%d")
        url = f"{BASE_URL}/{symbol}/range/5/minute/{day}/{day}"
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

        day_candles = [{
            "t": c["t"],       # timestamp (ms)
            "o": c["o"],       # open
            "h": c["h"],       # high
            "l": c["l"],       # low
            "c": c["c"],       # close
            "v": c["v"]        # volume
        } for c in data.get("results", [])]

        candles.extend(day_candles)

        days_back += 1

        if days_back > 30:
            break

    if len(candles) < input_len:
        raise Exception(f"Not enough candle data. Requested: {input_len}, Got: {len(candles)}")
    
    # Sort by timestamp (oldest first)
    candles_sorted = sorted(candles, key=lambda x: x["t"])

    # Get the last input_len candles
    last_candles = candles_sorted[-input_len:]

    # Print candles with EST time
    est_zone = ZoneInfo("America/New_York")
    for c in last_candles:
        est_time = datetime.datetime.fromtimestamp(c["t"] / 1000, tz=est_zone).strftime("%Y-%m-%d %H:%M:%S")
        print(f"Time(EST): {est_time}, O: {c['o']}, H: {c['h']}, L: {c['l']}, C: {c['c']}, V: {c['v']}")

    # return list(reversed(candles[-input_len:]))
    return last_candles