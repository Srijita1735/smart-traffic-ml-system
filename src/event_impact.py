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

def predict_event(hour, event_flag):
    temp_seq = last_seq.copy()

    
    temp_seq[-1][3] = hour / 23

    input_seq = torch.tensor(temp_seq.reshape(1, 24, 5), dtype=torch.float32)

    with torch.no_grad():
        pred_scaled = model(input_seq).item()

    
    full = np.zeros((1, 5))
    full[0][0] = pred_scaled
    traffic = scaler.inverse_transform(full)[0][0]

    
    if event_flag:
        traffic *= 1.4  # major surge

    
    if hour in [8, 9, 18, 19]:
        traffic *= 1.2

    return traffic


print("\n EVENT IMPACT ANALYSIS (IMPROVED):\n")

tests = [
    (9, False, "Normal Morning"),
    (9, True, "Event Morning"),
    (14, False, "Afternoon"),
    (18, True, "Festival Evening")
]

for hour, event, label in tests:
    traffic = predict_event(hour, event)
    print(f"{label}: Traffic = {traffic:.2f}")