import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib


data = pd.read_csv("data/master_dataset.csv")

print(" Dataset Loaded")


le_weather = LabelEncoder()
data["weather"] = le_weather.fit_transform(data["weather"])


features = [
    "traffic_volume",
    "avg_speed",
    "temperature",
    "rain_intensity",
    "event",
    "hour",
    "day_of_week"
]

X = data[features]
y = data["accident"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


model = XGBClassifier()
model.fit(X_train, y_train)


preds = model.predict(X_test)
print("Accident Accuracy:", accuracy_score(y_test, preds))

# Save
joblib.dump(model, "models/accident_model.pkl")

print("Accident model ready!")