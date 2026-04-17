rain = True
event = True
traffic = 480

print("\n🎯 DECISION SYSTEM:\n")

if rain and event:
    print("➡ Apply dynamic rerouting")
    print("➡ Increase green signal time")
elif traffic > 400:
    print("➡ Moderate congestion control")
else:
    print("➡ Normal operation")