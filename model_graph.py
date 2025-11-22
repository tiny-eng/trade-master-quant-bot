import pandas as pd
from sqlalchemy import create_engine
import torch
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt


# -----------------------------
# DEVICE
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# -----------------------------
# DB LOAD
# -----------------------------
DATABASE_URL = "postgresql://postgres:123456789@localhost/trader_master"
engine = create_engine(DATABASE_URL)

df = pd.read_sql("SELECT * FROM candle_a_minute;", engine)
df["time"] = pd.to_datetime(df["ts"], unit="s")
df = df.sort_values("time")

data = df["c"].values.reshape(-1, 1)
times = df["time"].values

# -----------------------------
# SCALE INPUTS
# -----------------------------
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# -----------------------------
# SETTINGS
# -----------------------------
INPUT_LEN = 60 * 5
PRED_INDEX = [1, 5, 15, 30, 60]   # Predict future 1m,5m,15m,30m,60m


# -----------------------------
# CREATE SEQUENCES (FIXED VERSION)
# -----------------------------
def create_sequence(data, input_len, pred_index, all_times):
    X, y, seq_times = [], [], []
    max_pred = max(pred_index)

    for i in range(len(data) - input_len - max_pred):
        X.append(data[i : i + input_len])

        y_sub = []
        for j in pred_index:
            y_sub.append(float(data[i + input_len + j - 1]))  # FIXED (scalar)

        y.append(y_sub)  # now shape = (5,)

        seq_times.append(all_times[i + input_len - 1])

    return np.array(X), np.array(y), np.array(seq_times)



X, y, seq_times = create_sequence(data_scaled, INPUT_LEN, PRED_INDEX, times)
print("X:", X.shape, "Y:", y.shape, "Times:", seq_times.shape)


# -----------------------------
# TRAIN/TEST SPLIT
# -----------------------------
split = int(len(X) * 0.95)

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
time_train, time_test = seq_times[:split], seq_times[split:]

X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).to(device)



# -----------------------------
# DATA LOADER
# -----------------------------
batch_size = 512
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


# -----------------------------
# MODEL
# -----------------------------
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=1, hidden_size=50, num_layers=1, batch_first=True)
        self.dropout1 = nn.Dropout(0.2)

        self.lstm2 = nn.LSTM(input_size=50, hidden_size=50, num_layers=1, batch_first=True)
        self.dropout2 = nn.Dropout(0.2)

        self.fc1 = nn.Linear(50, 25)
        self.fc2 = nn.Linear(25, len(PRED_INDEX))

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout1(x)

        x, _ = self.lstm2(x)
        x = self.dropout2(x)

        x = x[:, -1, :]
        x = self.fc1(x)
        x = self.fc2(x)
        return x


model = LSTMModel()
model.load_state_dict(torch.load("./model/new_3_model.pt", map_location=device))
model.eval()

with torch.no_grad():
    predictions = model(X_test).cpu().numpy()

# Convert (N,5) and (N,5)
pred_inv = scaler.inverse_transform(predictions)
y_test_inv = scaler.inverse_transform(y_test.cpu().numpy())


# -----------------------------
# PLOT EACH FUTURE PREDICTION
# -----------------------------
labels = ["+1 min", "+5 min", "+15 min", "+30 min", "+60 min"]

# plt.figure(figsize=(14, 12))

# for i in range(5):
#     plt.subplot(5, 1, i + 1)

#     plt.plot(time_test, y_test_inv[:, i], label=f"Real {labels[i]}")
#     plt.plot(time_test, pred_inv[:, i], label=f"Pred {labels[i]}", linestyle="--")

#     plt.title(f"Prediction: {labels[i]}")
#     plt.legend()
#     plt.tight_layout()

# plt.show()

plt.figure(figsize=(15, 1))

plt.plot(time_test, y_test[:, 1], label=f"Real Data")
plt.show()