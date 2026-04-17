print("\n DEMAND FORECAST:\n")

flows = {
    "Home → Office": 300,
    "Office → Home": 400
}

for k, v in flows.items():
    print(f"{k}: {v}")