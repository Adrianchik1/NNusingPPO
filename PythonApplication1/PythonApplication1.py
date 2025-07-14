import numpy as np
import nnfs
from nnfs.datasets import spiral_data

from activations import Activation_ReLU, Activatioin_Softmax
from weightsBiases import Layer_Dense
from losses import Loss_CategoricalCrossentrophy
from iterate import iteration
from logicManager import Optimiser

nnfs.init()
iterations = 0
loss = float('inf')
change = 0.05

dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()

dense2 = Layer_Dense(3, 3)
activation2 = Activatioin_Softmax()

X, y = spiral_data(samples=100, classes=3)

optimiser = Optimiser(X, y, activation1, activation2, change)

for i in range(0, 105):
    print(f"Iteration {iterations}")
    loss = iteration(X, y, dense1, dense2, activation1, activation2)
    loss, dense1, dense2 = optimiser.optimise(loss, dense1, dense2, iterations)
    print(f"Loss {loss}")
    iterations +=1
    