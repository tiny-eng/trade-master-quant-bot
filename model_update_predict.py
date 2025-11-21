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
DATABASE_URL = "postgresql://postgres:123456789@localhost/trader_master"

engine = create_engine(DATABASE_URL)

df = pd.read_sql("SELECT * FROM candle_a_minute;", engine)

df['time'] = pd.to_datetime(df['ts'], unit='s')

df = df.sort_values('time')

data = df['c'].values.reshape(-1, 1) # 2D input

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# INPUT_LEN = 60 * 24 * 15 # 1440 * 15 = 21600
INPUT_LEN = 60 * 3
 

PRED_INDEX = [1, 5, 15, 30, 60]


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

MODEL_PATH = "./model/new_model.pt"
torch.save(model.state_dict(), MODEL_PATH)

print(f"Model saved to {MODEL_PATH}")

last_seq = data_scaled[-INPUT_LEN:]
last_seq = torch.tensor(last_seq, dtype=torch.float32).unsqueeze(0).to(device)

future_scaled = model(last_seq).detach().cpu().numpy().reshape(-1, 1)
future = scaler.inverse_transform(future_scaled)

print(future)