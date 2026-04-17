import pandas as pd
import numpy as np

# Load dataset
data = pd.read_csv("data/traffic_real.csv")

# Convert datetime
data["datetime"] = pd.to_datetime(data["date_time"])

# -----------------------------
# CREATE TRAFFIC VOLUME (IMPORTANT)
# -----------------------------
# Using weather + time patterns

data["hour"] = data["datetime"].dt.hour
data["day_of_week"] = data["datetime"].dt.dayofweek

# Base traffic pattern
data["traffic_volume"] = (
    200 +
    data["hour"] * 10 +
    (data["day_of_week"] * 5) -
    (data["rain_p_h"] * 20) +
    np.random.randint(-30, 30, len(data))
)

# -----------------------------
# ADD OTHER FEATURES
# -----------------------------

data["avg_speed"] = 80 - (data["traffic_volume"] * 0.1)

data["event"] = data["hour"].isin([8, 9, 18, 19]).astype(int)

data["accident"] = ((data["rain_p_h"] > 0.5) & (data["traffic_volume"] > 300)).astype(int)

data["vehicle_type"] = np.random.choice(["car", "bike", "bus"], len(data))

data["location_id"] = 1

# Congestion
data["congestion_level"] = pd.cut(
    data["traffic_volume"],
    bins=[0, 200, 400, 1000],
    labels=["Low", "Medium", "High"]
)

# Final columns
final_cols = [
    "datetime",
    "traffic_volume",
    "avg_speed",
    "weather_type",
    "temperature",
    "rain_p_h",
    "event",
    "accident",
    "vehicle_type",
    "hour",
    "day_of_week",
    "congestion_level",
    "location_id"
]

data = data[final_cols]

# Rename for consistency
data = data.rename(columns={
    "weather_type": "weather",
    "rain_p_h": "rain_intensity"
})

# Save final dataset
data.to_csv("data/master_dataset.csv", index=False)

print("✅ FINAL REAL DATASET READY!")