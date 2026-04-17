areas = {
    "Area A": 450,
    "Area B": 300,
    "Area C": 500
}

print("\n⚖️ TRAFFIC FAIRNESS:\n")

for k, v in areas.items():
    print(f"{k}: {v}")

print("\n⚠️ Area C consistently high → needs optimization")