import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import networkx as nx
import joblib
import random


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
input_seq = torch.tensor(last_seq.reshape(1, 24, 5), dtype=torch.float32)

with torch.no_grad():
    pred_scaled = model(input_seq).item()


full = np.zeros((1, 5))
full[0][0] = pred_scaled
base_traffic = scaler.inverse_transform(full)[0][0]

print(f" Base Traffic Level: {base_traffic:.2f}")


G = nx.Graph()

def road_cost(base_distance, traffic):
    return base_distance * (1 + traffic / 500)


edges = [
    ("A", "B", 5),
    ("A", "C", 6),
    ("B", "D", 4),
    ("C", "D", 7),
    ("C", "E", 5),
    ("D", "E", 3)
]

for u, v, dist in edges:
   
    traffic_variation = base_traffic * random.uniform(0.7, 1.3)
    cost = road_cost(dist, traffic_variation)
    
    G.add_edge(u, v, weight=cost)
    
    print(f"🛣 Road {u}-{v}: Traffic={traffic_variation:.2f}, Cost={cost:.2f}")


start = "A"
end = "E"

path = nx.dijkstra_path(G, start, end, weight="weight")
cost = nx.dijkstra_path_length(G, start, end, weight="weight")

print("\n Best Route:", path)
print("Total Cost:", round(cost, 2))