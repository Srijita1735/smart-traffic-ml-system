import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import joblib


data = pd.read_csv("data/master_dataset.csv")
data["datetime"] = pd.to_datetime(data["datetime"])
data = data.sort_values("datetime")


features = [
    "traffic_volume",
    "temperature",
    "rain_intensity",
    "hour",
    "day_of_week"
]

values = data[features].values


scaler = joblib.load("models/scaler.pkl")
scaled = scaler.transform(values)


class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=5, hidden_size=64, batch_first=True)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

model = LSTMModel()
model.load_state_dict(torch.load("models/traffic_lstm.pth"))
model.eval()


def create_sequences(data, seq_length=24):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length][0])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled)

X = torch.tensor(X, dtype=torch.float32)


with torch.no_grad():
    preds = model(X).numpy()

preds_full = np.zeros((len(preds), 5))
preds_full[:, 0] = preds.flatten()

actual_full = np.zeros((len(y), 5))
actual_full[:, 0] = y

preds_inv = scaler.inverse_transform(preds_full)[:, 0]
actual_inv = scaler.inverse_transform(actual_full)[:, 0]


plt.figure()
plt.plot(actual_inv[:300], label="Actual")
plt.plot(preds_inv[:300], label="Predicted")
plt.title("Actual vs Predicted Traffic (Final)")
plt.xlabel("Time")
plt.ylabel("Traffic Volume")
plt.legend()
plt.show()


last_seq = scaled[-24:]
current_seq = torch.tensor(last_seq.reshape(1, 24, 5), dtype=torch.float32)

future = []

for _ in range(48):
    with torch.no_grad():
        pred = model(current_seq)
    
    future.append(pred.item())

    new = current_seq.numpy()
    
  
    next_step = new[:, -1, :].copy()
    next_step[0][0] = pred.item()
    
    new = np.append(new[:, 1:, :], next_step.reshape(1, 1, 5), axis=1)
    
    current_seq = torch.tensor(new, dtype=torch.float32)

future_full = np.zeros((48, 5))
future_full[:, 0] = future

future_inv = scaler.inverse_transform(future_full)[:, 0]

plt.figure()
plt.plot(future_inv, label="Future Traffic (48 hrs)")
plt.title("Future Traffic Prediction (Final)")
plt.xlabel("Hours Ahead")
plt.ylabel("Traffic Volume")
plt.legend()
plt.show()
