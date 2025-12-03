from datetime import datetime
from zoneinfo import ZoneInfo

def format_response(symbol, start_time, start_close ,last_time, last_close, predictions):

    est_zone = ZoneInfo("America/New_York")
    start_est_dt = datetime.fromtimestamp(start_time / 1000, tz=est_zone)
    last_est_dt = datetime.fromtimestamp(last_time / 1000, tz=est_zone)


    start_readable_time = start_est_dt.strftime("%Y-%m-%d %H:%M:%S")
    last_readable_time = last_est_dt.strftime("%Y-%m-%d %H:%M:%S")

    start_rounded_close = round(start_close, 2)
    last_rounded_close = round(last_close, 2)
    rounded_predictions = [round(p, 2) for p in predictions]



    return {
        "symbol": symbol,
        "start_time": start_readable_time,
        "start_close": start_rounded_close,
        "last_time": last_readable_time,
        "last_close": last_rounded_close,
        "prediction": rounded_predictions,
    }