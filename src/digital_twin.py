print("\n DIGITAL TWIN SIMULATION:\n")

scenarios = {
    "Normal": 350,
    "Rain": 420,
    "Event": 500
}

for k, v in scenarios.items():
    print(f"{k} Scenario → Traffic: {v}")