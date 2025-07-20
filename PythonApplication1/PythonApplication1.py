import numpy as np
import nnfs
from nnfs.datasets import spiral_data

from activations import Activation_ReLU, Activatioin_Softmax
from weightsBiases import Layer_Dense
from iterate import iteration
from logicManager import Optimiser
from charts import makeChart

nnfs.init()
loss = float('inf')
iterations = 1000
change = 0.05
losses = []         #array to store the last 10 losses
progressOfLosses = []       #an array to store all losses(for future analysis)
differenceOfLosses = []
implementBiases = False

dense1 = Layer_Dense(2, 10)
activation1 = Activation_ReLU()

dense2 = Layer_Dense(10, 3)
activation2 = Activatioin_Softmax()

X, y = spiral_data(samples=100, classes=3)

denses = [dense1, dense2]
activations = [activation1, activation2]

optimiser = Optimiser(X, y, activations, change)

for i in range(0, iterations):
    print(f"Iteration {i}")
    loss = iteration(X, y, denses, activations)
    loss, denses = optimiser.optimise(loss, denses)
    print(f"Loss {loss}")

    if len(losses) > 13: losses.pop(0)                                      # removes the oldest loss if it has more than 10 elements
    if len(losses) > 10 and np.allclose(losses, losses[0]) == True:         # if loss hasnt changed in the last 10 iterations implements biases
        break
    if len(losses)>2:
        differenceOfLosses.append(losses[0] - losses[1])  # appends the change in loss to the progressOfLosses array)
        progressOfLosses.append(loss)


print(loss)
print(iterations)
makeChart(progressOfLosses)
makeChart(differenceOfLosses)
input("Press enter to exit")