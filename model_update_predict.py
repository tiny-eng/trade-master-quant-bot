import pandas as pd
from sqlalchemy import create_engine
import torch
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device: ", device )

# Database connection info
DATABASE_URL = "postgresql://postgres:123456789@localhost/trader_master"

engine = create_engine(DATABASE_URL)

df = pd.read_sql("SELECT * FROM candle_a_minute;", engine)

df['time'] = pd.to_datetime(df['ts'], unit='s')

df = df.sort_values('time')

data = df['c'].values.reshape(-1, 1) # 2D input
time = df['time'].values.reshape(-1, 1)

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# INPUT_LEN = 60 * 24 * 15 # 1440 * 15 = 21600
INPUT_LEN = 60 * 5
PRED_INDEX = [1, 5, 15, 30, 60]

def create_sequence(data, input_len, pred_index):
    X, y = [], []
    for i in range(len(data) - input_len - 60):
        X.append(data[i: i + input_len])

        y_sub = []
        for j in pred_index:
            y_sub.append(data[i + input_len + j - 1])

        y.append(np.array(y_sub).reshape(-1))

    return np.array(X), np.array(y).reshape(len(y), len(pred_index))

X, y = create_sequence(data_scaled, INPUT_LEN, PRED_INDEX)
print("X:", X.shape, "Y:", y.shape)

# Train/test split
split = int(len(X) * 0.95)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

batch_size = 512

class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()

        self.lstm1 = nn.LSTM(
            input_size=1,
            hidden_size=50,
            num_layers=1,
            batch_first=True
        )

        self.dropout1 = nn.Dropout(0.2)

        self.lstm2 = nn.LSTM(
            input_size=50,
            hidden_size=50,
            num_layers=1,
            batch_first=True
        )

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

model = LSTMModel().to(device)

model = LSTMModel()
model.load_state_dict(torch.load("./model/new_3_model.pt", map_location=device))
model.eval()

with torch.no_grad():
    y_pred = model(X_test)

print("Pred shape:", y_pred.shape)
print("True shape:", y_test.shape)

mse = torch.mean((y_pred - y_test) ** 2).item()

print("MSE:", mse)

mse_per_index = {}

for i, h in enumerate(PRED_INDEX):
    mse_h = torch.mean((y_pred[:, i] - y_test[:, i]) ** 2).item()
    mse_per_index[h] = mse_h

print("MSE per index")
for h, val in mse_per_index.items():
    print(f" {h} min: {val}")

y_test_np = y_test.cpu().numpy()
y_pred_np = y_pred.cpu().numpy()

y_test_inv = scaler.inverse_transform(y_test_np)
y_pred_inv = scaler.inverse_transform(y_pred_np)


plt.figure(figsize=(14, 12))

for i, h in enumerate(PRED_INDEX):
    plt.subplot(len(PRED_INDEX), 1, i+1)
    plt.plot(y_test_inv[:, i], label="True")
    plt.plot(y_pred_inv[:, i], label="Predicted")
    plt.title(f"{h}-minute Prediction")
    plt.legend()

plt.tight_layout()
plt.show()