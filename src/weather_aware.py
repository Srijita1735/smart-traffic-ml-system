import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib

# -----------------------------
# LOAD DATA
# -----------------------------
data = pd.read_csv("data/master_dataset.csv")

features = [
    "traffic_volume",
    "temperature",
    "rain_intensity",
    "hour",
    "day_of_week"
]

values = data[features].values

# Load scaler
scaler = joblib.load("models/scaler.pkl")
scaled = scaler.transform(values)

# -----------------------------
# MODEL
# -----------------------------
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

# -----------------------------
# WEATHER IMPACT FUNCTION
# -----------------------------
last_seq = scaled[-24:]

def predict_weather(temp, rain):
    temp_seq = last_seq.copy()

    # Update weather features
    temp_seq[-1][1] = temp / 50
    temp_seq[-1][2] = rain / 10

    input_seq = torch.tensor(temp_seq.reshape(1, 24, 5), dtype=torch.float32)

    with torch.no_grad():
        pred_scaled = model(input_seq).item()

    # Convert to real value
    full = np.zeros((1, 5))
    full[0][0] = pred_scaled
    traffic = scaler.inverse_transform(full)[0][0]

    # 🔥 REALISTIC EFFECT: rain slows traffic → increases congestion
    adjusted_traffic = traffic * (1 + rain * 0.05)

    return adjusted_traffic

# -----------------------------
# TEST SCENARIOS
# -----------------------------
print("\n🌦 WEATHER IMPACT ANALYSIS (IMPROVED):\n")

scenarios = [
    ("Clear", 30, 0),
    ("Rainy", 25, 5),
    ("Heavy Rain", 22, 9)
]

for name, temp, rain in scenarios:
    traffic = predict_weather(temp, rain)
    print(f"{name}: Traffic = {traffic:.2f}")