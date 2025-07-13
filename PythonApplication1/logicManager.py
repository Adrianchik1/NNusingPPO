import numpy as np
import copy
from iterate import iteration

def getWeightCords(weights, iteration):
    d, r, c = len(weights), len(weights[0]), len(weights[0][0])
    iteration %= d * r * c
    return [iteration // (r * c), (iteration // c) % r, iteration % c]

def changeWeight(weights, iteration, change):
    weightCords = getWeightCords(weights, iteration)
    weights[weightCords[0]][weightCords[1]][weightCords[2]] += change
    return weights[0], weights[1]

def optimise(X, y, loss, dense1, dense2, iterations, activation1, activation2):
    positiveDense1, positiveDense2 = copy.deepcopy(dense1), copy.deepcopy(dense2)
    negativeDense1, negativeDense2 = copy.deepcopy(dense1), copy.deepcopy(dense2)

    positiveDense1.weights, positiveDense2.weights = changeWeight([copy.deepcopy(dense1.weights), copy.deepcopy(dense2.weights)], iterations, 0.05)
    negativeDense1.weights, negativeDense2.weights = changeWeight([copy.deepcopy(dense1.weights), copy.deepcopy(dense2.weights)], iterations, -0.05)

    positiveLoss = iteration(X, y, positiveDense1, positiveDense2, activation1, activation2)
    negativeLoss = iteration(X, y, negativeDense1, negativeDense2, activation1, activation2)

    losses = [(positiveLoss, positiveDense1, positiveDense2),
              (negativeLoss, negativeDense1, negativeDense2),
              (loss, dense1, dense2)]
    return min(losses, key=lambda x: x[0]) 
