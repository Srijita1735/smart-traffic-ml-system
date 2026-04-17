import shap
import pandas as pd
import joblib
import numpy as np

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

X = data[features]

# -----------------------------
# LOAD SCALER
# -----------------------------
scaler = joblib.load("models/scaler.pkl")
X_scaled = scaler.transform(X)

# -----------------------------
# LOAD MODEL
# -----------------------------
model = joblib.load("models/congestion_model.pkl")

# -----------------------------
# SHAP EXPLAINER (FINAL FIX)
# -----------------------------
explainer = shap.Explainer(model, X_scaled)
shap_values = explainer(X_scaled)

# -----------------------------
# EXTRACT FIRST SAMPLE
# -----------------------------
sample = shap_values.values[0]

# Safety check (avoid mismatch)
sample = np.array(sample).flatten()
used_features = features[:len(sample)]

# -----------------------------
# PRINT SHAP VALUES
# -----------------------------
print("\nEXPLAINABLE AI OUTPUT:\n")

for f, val in zip(used_features, sample):
    print(f"{f}: contribution {val:.4f}")

# -----------------------------
# INTERPRETATION
# -----------------------------
print("\nINTERPRETATION:\n")

feature_dict = dict(zip(used_features, sample))

if feature_dict.get("traffic_volume", 0) > 0:
    print("High traffic volume contributes to congestion")

if feature_dict.get("rain_intensity", 0) > 0:
    print("Rainfall increases congestion")

if feature_dict.get("hour", 0) > 0:
    print("Peak hour contributes to traffic increase")

if feature_dict.get("temperature", 0) > 0:
    print("Temperature affects traffic flow")

if all(v <= 0 for v in feature_dict.values()):
    print("Traffic conditions are stable")