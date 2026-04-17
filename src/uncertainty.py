import numpy as np

predictions = [340, 360, 355, 345, 350]

mean = np.mean(predictions)
std = np.std(predictions)

print("\n📊 UNCERTAINTY ESTIMATION:\n")
print(f"Traffic Prediction: {mean:.2f} ± {std:.2f}")