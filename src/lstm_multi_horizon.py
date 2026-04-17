import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
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


scaler = MinMaxScaler()
scaled = scaler.fit_transform(values)


def create_sequences(data, seq_length=24, horizon=48):
    X, y = [], []
    for i in range(len(data) - seq_length - horizon):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length:i+seq_length+horizon, 0])  # only traffic
    return np.array(X), np.array(y)

X, y = create_sequences(scaled)

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)


split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]


class MultiLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=5, hidden_size=64, batch_first=True)
        self.fc = nn.Linear(64, 48)  # output 48 steps

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

model = MultiLSTM()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


epochs = 30

for epoch in range(epochs):
    model.train()
    
    output = model(X_train)
    loss = criterion(output, y_train)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")


torch.save(model.state_dict(), "models/multi_lstm.pth")
joblib.dump(scaler, "models/multi_scaler.pkl")

print(" Multi-horizon LSTM trained!")