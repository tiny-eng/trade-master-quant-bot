import pandas as pd
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Database connection info
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

def create_sequences(data, seq_len, pred_len):
    X, y = [], []
    for i in range(len(data) - seq_len - pred_len):
        X.append(data[i : i + seq_len])
        y.append(data[i + seq_len : i + seq_len + pred_len])

    return np.array(X), np.array(y)

X, y = create_sequences(data_scaled, SEQ_LEN, PRED_LEN)
print("X:", X.shape, "Y:", y.shape)

# Train/test split
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

batch_size = 512

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# LSTM model
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
    
model = LSTMModel().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training
EPOCHS = 30

for epoch in range(EPOCHS):
    model.train()

    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()

        y_pred = model(batch_x)
        loss = criterion(y_pred, batch_y.reshape(y_pred.shape))
        loss.backward()
        optimizer.step()

    if epoch % 2 == 0:
        print(f"Epoch {epoch}/{EPOCHS} Loss: {loss.item():.6f}")

# Predict
model.eval()

MODEL_PATH = "./model/lstm_model.pt"
torch.save(model.state_dict(), MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")


last_seq = data_scaled[-SEQ_LEN:]
last_seq = torch.tensor(last_seq, dtype=torch.float32).unsqueeze(0).to(device)

future_scaled = model(last_seq).detach().cpu().numpy().reshape(-1, 1)
future = scaler.inverse_transform(future_scaled)

print("\nPredicted next 60 minutes:")
print(future)

# Plot results

plt.figure(figsize=(12, 5))
plt.plot(future, label="Predicted 60 min future")
plt.title("LSTM Next 60 Minutes Prediction")
plt.xlabel("Minutes Ahead")
plt.ylabel("Price")
plt.legend()
plt.show()



