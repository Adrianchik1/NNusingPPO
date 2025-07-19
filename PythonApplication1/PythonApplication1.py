import numpy as np
import nnfs
from nnfs.datasets import spiral_data

from activations import Activation_ReLU, Activatioin_Softmax
from weightsBiases import Layer_Dense
from iterate import iteration
from logicManager import Optimiser
from charts import makeChart

nnfs.init()
iterations = 0
loss = float('inf')
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

optimiser = Optimiser(X, y, activation1, activation2, change)

for i in range(0, 200):
    print(f"Iteration {iterations}")
    loss = iteration(X, y, dense1, dense2, activation1, activation2)
    loss, dense1, dense2 = optimiser.optimise(loss, dense1, dense2)
    print(f"Loss {loss}")

    losses.append(loss)
    #progressOfLosses.append(loss)
    if len(losses) > 13: losses.pop(0)                                      # removes the oldest loss if it has more than 10 elements
    if len(losses) > 10 and np.allclose(losses, losses[0]) == True:         # if loss hasnt changed in the last 10 iterations implements biases
        #if implementBiases == True:                                         # if biases have already been implemented, then break
        #     break
        #implementBiases = True
        progress = iterations - 10
        break
    if len(losses)>2:
        differenceOfLosses.append(losses[0] - losses[1])  # appends the change in loss to the progressOfLosses array)
        progressOfLosses.append(loss)
    # if len(losses)>2 and losses[0] - losses[-1] < 0.001:
    #     optimiser.change /= 2


    iterations +=1
    


print(loss)
print(iterations)
makeChart(progressOfLosses)
makeChart(differenceOfLosses)
input("Press enter to exit")