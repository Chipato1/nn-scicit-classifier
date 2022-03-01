import numpy as np

x = np.array([0.05, 0.1])
weights = np.array([[0.15, 0.25], [0.2, 0.30]])
biases = np.array([0.35, 0.35])
print(weights)
print(x)

print(x @ weights + biases)