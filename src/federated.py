import numpy as np


city1 = np.random.rand(5)
city2 = np.random.rand(5)
city3 = np.random.rand(5)


global_model = (city1 + city2 + city3) / 3

print("\n FEDERATED LEARNING:\n")
print("City1:", city1)
print("City2:", city2)
print("City3:", city3)

print("\nGlobal Model:", global_model)