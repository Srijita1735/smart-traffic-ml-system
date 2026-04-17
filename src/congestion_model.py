import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
data = pd.read_csv("data/master_dataset.csv")

print("✅ Dataset Loaded")

# Encode categorical
le_weather = LabelEncoder()
data["weather"] = le_weather.fit_transform(data["weather"])

le_congestion = LabelEncoder()
data["congestion_level"] = le_congestion.fit_transform(data["congestion_level"])

# Features
features = [
    "traffic_volume",
    "avg_speed",
    "temperature",
    "rain_intensity",
    "hour",
    "day_of_week"
]

X = data[features]
y = data["congestion_level"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
print("🚦 Congestion Accuracy:", accuracy_score(y_test, preds))

# Save
joblib.dump(model, "models/congestion_model.pkl")

print("✅ Congestion model ready!")