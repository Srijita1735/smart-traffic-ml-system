import pandas as pd
import numpy as np

data = pd.read_csv("data/traffic_real.csv")


data["datetime"] = pd.to_datetime(data["date_time"])



data["hour"] = data["datetime"].dt.hour
data["day_of_week"] = data["datetime"].dt.dayofweek


data["traffic_volume"] = (
    200 +
    data["hour"] * 10 +
    (data["day_of_week"] * 5) -
    (data["rain_p_h"] * 20) +
    np.random.randint(-30, 30, len(data))
)



data["avg_speed"] = 80 - (data["traffic_volume"] * 0.1)

data["event"] = data["hour"].isin([8, 9, 18, 19]).astype(int)

data["accident"] = ((data["rain_p_h"] > 0.5) & (data["traffic_volume"] > 300)).astype(int)

data["vehicle_type"] = np.random.choice(["car", "bike", "bus"], len(data))

data["location_id"] = 1


data["congestion_level"] = pd.cut(
    data["traffic_volume"],
    bins=[0, 200, 400, 1000],
    labels=["Low", "Medium", "High"]
)


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


data = data.rename(columns={
    "weather_type": "weather",
    "rain_p_h": "rain_intensity"
})


data.to_csv("data/master_dataset.csv", index=False)

print(" FINAL REAL DATASET READY!")