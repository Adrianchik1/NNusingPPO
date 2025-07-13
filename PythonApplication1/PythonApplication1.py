import numpy as np
import nnfs
from nnfs.datasets import spiral_data

from activations import Activation_ReLU, Activatioin_Softmax
from weightsBiases import Layer_Dense
from losses import Loss_CategoricalCrossentrophy
from iterate import iteration
from logicManager import optimise

nnfs.init()
iterations = 0
loss = float('inf')

dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()

dense2 = Layer_Dense(3, 3)
activation2 = Activatioin_Softmax()

X, y = spiral_data(samples=100, classes=3)

for i in range(1, 100000):
    print(f"Iteration {iterations}")
    loss = iteration(X, y, dense1, dense2, activation1, activation2)
    loss, dense1, dense2 = optimise(X, y, loss, dense1, dense2, iterations, activation1, activation2)
    print(f"Loss {loss}")
    iterations +=1
    