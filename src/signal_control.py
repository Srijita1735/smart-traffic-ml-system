import numpy as np


states = ["low", "medium", "high"]


actions = [10, 30, 60]

Q = np.zeros((3,3))


for _ in range(1000):
    state = np.random.randint(0,3)
    
    action = np.argmax(Q[state])
    
    if state == 2 and action == 2:
        reward = 10
    elif state == 0 and action == 0:
        reward = 8
    else:
        reward = -5
    
    Q[state][action] += 0.1 * (reward + 0.9*np.max(Q[state]) - Q[state][action])

print("\n SMART SIGNAL CONTROL:\n")

for i, s in enumerate(states):
    best = actions[np.argmax(Q[i])]
    print(f"{s} traffic → green time: {best} sec")