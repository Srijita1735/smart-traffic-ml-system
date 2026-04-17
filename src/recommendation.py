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
        self.lstm = nn.LSTM(input_size=5, hidden_size=64, batch_first=True)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


model = LSTMModel()
model.load_state_dict(torch.load("models/traffic_lstm.pth"))
model.eval()


last_seq = scaled[-24:]


def predict(hour, rain=3, event=False):
    temp_seq = last_seq.copy()

    
    temp_seq[-1][3] = hour / 23
    temp_seq[-1][2] = rain / 10

    input_seq = torch.tensor(temp_seq.reshape(1, 24, 5), dtype=torch.float32)

    with torch.no_grad():
        pred_scaled = model(input_seq).item()

    
    full = np.zeros((1, 5))
    full[0][0] = pred_scaled
    traffic = scaler.inverse_transform(full)[0][0]

    
    if hour in [8, 9, 18, 19]:
        traffic *= 1.6

    
    elif hour in [7, 10, 17, 20]:
        traffic *= 1.3

    
    elif hour in [11, 12, 13, 14]:
        traffic *= 0.85

    
    elif hour in [21, 22]:
        traffic *= 0.9

    
    if rain > 5:
        traffic *= 1.25

    
    if event:
        traffic *= 1.5

    return traffic


results = []

for hour in range(6, 22):
    traffic = predict(hour, rain=3, event=False)
    results.append((hour, traffic))


results = sorted(results, key=lambda x: x[1])


print("\n SMART TRAVEL RECOMMENDATION (ULTIMATE):\n")

for hour, traffic in results[:5]:
    print(f" {hour}:00 → Expected Traffic: {traffic:.2f}")

print("\n⚠️ AVOID THESE TIMES:\n")

for hour, traffic in results[-3:]:
    print(f" {hour}:00 → Traffic: {traffic:.2f}")