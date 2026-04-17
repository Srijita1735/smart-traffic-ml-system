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


scaler = joblib.load("models/multi_scaler.pkl")
scaled = scaler.transform(values)


class MultiLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=5, hidden_size=64, batch_first=True)
        self.fc = nn.Linear(64, 48)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

model = MultiLSTM()
model.load_state_dict(torch.load("models/multi_lstm.pth"))
model.eval()


last_seq = scaled[-24:]
input_seq = torch.tensor(last_seq.reshape(1, 24, 5), dtype=torch.float32)


with torch.no_grad():
    future = model(input_seq).numpy().flatten()


full = np.zeros((48, 5))
full[:, 0] = future

future_inv = scaler.inverse_transform(full)[:, 0]


plt.figure()
plt.plot(future_inv)
plt.title("Multi-Horizon Prediction (48 steps)")
plt.xlabel("Hours Ahead")
plt.ylabel("Traffic Volume")
plt.show()