import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib


data = pd.read_csv("data/master_dataset.csv")

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
        self.lstm = nn.LSTM(5, 64, batch_first=True)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

model = LSTMModel()
model.load_state_dict(torch.load("models/traffic_lstm.pth"))
model.eval()


last_seq = scaled[-24:]
input_seq = torch.tensor(last_seq.reshape(1, 24, 5), dtype=torch.float32)

with torch.no_grad():
    pred_scaled = model(input_seq).item()


full = np.zeros((1, 5))
full[0][0] = pred_scaled
base_traffic = scaler.inverse_transform(full)[0][0]


car = base_traffic * 0.6
bike = base_traffic * 0.25
bus = base_traffic * 0.15


print("\n🚦 MULTI-MODAL TRAFFIC (FINAL):\n")
print(f"Cars: {car:.2f}")
print(f"Bikes: {bike:.2f}")
print(f"Buses: {bus:.2f}")