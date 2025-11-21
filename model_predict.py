import torch
import numpy as np
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
from sqlalchemy import create_engine
import pandas as pd
import matplotlib.pyplot as plt 

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Make sequences
SEQ_LEN = 180
PRED_LEN = 60

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=PRED_LEN):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# --- Load model ---
model = LSTMModel()
model.load_state_dict(torch.load("./model/lstm_model.pt", map_location=device))
model.eval()

# --- Prepare last 60 minutes ---
DATABASE_URL = "postgresql://postgres:123456789@localhost/trader_master"

engine = create_engine(DATABASE_URL)

# Load table
df = pd.read_sql("SELECT * FROM candle_a_minute;", engine)

# Convert timestamp
df['time'] = pd.to_datetime(df['ts'], unit='s')

df = df.sort_values('time')

# Use only close price
data = df['c'].values.reshape(-1, 1)

# Scale data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Make sequences
SEQ_LEN = 180
PRED_LEN = 60


# --- Prepare last SEQ_LEN minutes ---
last_seq = data_scaled[-SEQ_LEN:]
last_seq_tensor = torch.tensor(last_seq, dtype=torch.float32).unsqueeze(0).to(device)

# Inverse last real values
last_real = scaler.inverse_transform(last_seq)

# --- Predict ---
with torch.no_grad():
    future_scaled = model(last_seq_tensor).cpu().numpy().reshape(-1, 1)
    future = scaler.inverse_transform(future_scaled)

# --- Build time axis ---
# last SEQ_LEN timestamps
time_real = df['time'].iloc[-SEQ_LEN:]

# predicted timestamps → each 1 min after last real timestamp
last_time = time_real.iloc[-1]
time_future = pd.date_range(start=last_time, periods=PRED_LEN + 1, freq="1min")[1:]

# --- Plot ---
plt.figure(figsize=(12, 5))

plt.plot(time_real, last_real, label=f"Last {SEQ_LEN} Real Prices")
plt.plot(time_future, future, label=f"Next {PRED_LEN} Predicted Prices")

plt.title("Real Price + LSTM Forecast")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
