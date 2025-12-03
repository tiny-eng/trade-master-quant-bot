import os
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

from app.config.constants import DEVICE, PRED_INDEX

MODEL_CACHE = {}

class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()

        self.lstm1 = nn.LSTM(
            input_size = 5,
            hidden_size = 64,
            num_layers = 1,
            batch_first = True
        )

        self.dropout1 = nn.Dropout(0.2)

        self.lstm2 = nn.LSTM(
            input_size = 64,
            hidden_size = 64,
            num_layers = 1,
            batch_first = True
        )

        self.dropout2 = nn.Dropout(0.2)

        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 12)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout1(x)

        x, _ = self.lstm2(x)
        x = self.dropout2(x)

        x = x[:, -1, :]
        x = self.fc1(x)
        x = self.fc2(x)

        return x


def load_model(symbol: str):
    if symbol in MODEL_CACHE:
        return MODEL_CACHE[symbol]

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(base, "model", f"{symbol}_model.pt")

    if not os.path.exists(path):
        raise FileNotFoundError(f"No model found for symbol {symbol}, {path}")

    model = LSTMModel().to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()

    MODEL_CACHE[symbol] = model
    return model

def load_scaler():
    scaler = MinMaxScaler()
    return scaler
