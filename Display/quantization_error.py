import matplotlib.pyplot as plt

import pickle as pkl

import numpy as np

with open("../Testing/distances.pkl", "rb") as file:
    distances = pkl.load(file)

print(distances, distances.size)

print(distances)

# (there are exactly 102500 points in the "good" training data)
x = np.array([(i * 512 + 102500) for i in range(distances.size)])
y = np.array(distances)

plt.plot(x, y)

plt.ticklabel_format(style='plain', axis='x')

plt.ylim(0, 0.0020)

plt.xlabel("Sequence Number")
plt.ylabel("Quantization Error")
plt.title("Graph of the Quantization Error of all sequences to Good Map")

plt.grid(True)

plt.show()

