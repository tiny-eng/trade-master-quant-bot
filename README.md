# Trading MASTER QUANT BOT

Python 3.10.0

python -m venv .venv

.venv\Scripts\activate

pip install -r requirements. txt

python -m uvicorn app.main:app --reload --port 8000


http://127.0.0.1:8000/predict/live?symbol=nvda&target=5