import pandas as pd
import numpy as np


data = pd.read_csv("data/master_dataset.csv")

traffic = data["traffic_volume"]

mean = traffic.mean()
std = traffic.std()

threshold = mean + 2 * std

print("\n ANOMALY DETECTION:\n")

for i, val in enumerate(traffic[-50:]):
    if val > threshold:
        print(f" Anomaly detected at index {i}: Traffic = {val}")