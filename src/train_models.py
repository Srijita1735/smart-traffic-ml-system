import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
data = pd.read_csv("data/master_dataset.csv")

print("✅ Dataset Loaded")


le_weather = LabelEncoder()
le_vehicle = LabelEncoder()
le_congestion = LabelEncoder()

data["weather"] = le_weather.fit_transform(data["weather"])
data["vehicle_type"] = le_vehicle.fit_transform(data["vehicle_type"])
data["congestion_level"] = le_congestion.fit_transform(data["congestion_level"])


features = [
    "traffic_volume",
    "avg_speed",
    "weather",
    "temperature",
    "rain_intensity",
    "event",
    "hour",
    "day_of_week"
]

X = data[features]


y_congestion = data["congestion_level"]

X_train, X_test, y_train, y_test = train_test_split(X, y_congestion, test_size=0.2)

rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train, y_train)

preds = rf_model.predict(X_test)
print("🚦 Congestion Accuracy:", accuracy_score(y_test, preds))

joblib.dump(rf_model, "models/congestion_model.pkl")


y_accident = data["accident"]

X_train, X_test, y_train, y_test = train_test_split(X, y_accident, test_size=0.2)

xgb_model = XGBClassifier()
xgb_model.fit(X_train, y_train)

preds = xgb_model.predict(X_test)
print("Accident Accuracy:", accuracy_score(y_test, preds))

joblib.dump(xgb_model, "models/accident_model.pkl")

print(" Models trained successfully!")