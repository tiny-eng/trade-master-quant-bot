import pandas as pd
from sqlalchemy import create_engine
import torch
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device: ", device )

# Database connection info
DATABASE_URL = "postgresql://postgres:123456@localhost/trader_master"

engine = create_engine(DATABASE_URL)

df = pd.read_sql("SELECT * FROM candle_spy_minute;", engine)

df['time'] = pd.to_datetime(df['ts'], unit='s')

df = df.sort_values('time')

data = df['c'].values.reshape(-1, 1) # 2D input

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

X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

batch_size = 512

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

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
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training
EPOCHS = 10

for epoch in range(EPOCHS):
    model.train()

    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()

        y_pred = model(batch_x)
        loss = criterion(y_pred, batch_y)
        loss.backward()
        optimizer.step()


    print(f"Epoch {epoch}/{EPOCHS} Loss: {loss.item():.6f}")

# Predict
model.eval()

MODEL_PATH = "./model/spy_model.pt"
torch.save(model.state_dict(), MODEL_PATH)

print(f"Model saved to {MODEL_PATH}")

last_seq = data_scaled[-INPUT_LEN:]
last_seq = torch.tensor(last_seq, dtype=torch.float32).unsqueeze(0).to(device)

future_scaled = model(last_seq).detach().cpu().numpy().reshape(-1, 1)
future = scaler.inverse_transform(future_scaled)

print(future)