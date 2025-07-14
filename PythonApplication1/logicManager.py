import numpy as np
import copy
from iterate import iteration

class Optimiser():
    def __init__(self, X, y, activation1, activation2, change):
        self.X = X
        self.y = y
        self.activation1 = activation1
        self.activation2 = activation2
        self.change = change

    def getWeightCords(self, weights, nthElem):
        d, r, c = len(weights), len(weights[0]), len(weights[0][0])
        nthElem %= d * r * c
        return [nthElem // (r * c), (nthElem // c) % r, nthElem % c]

    def changeWeight(self, weights, nthElem, change):
        weightCords = self.getWeightCords(weights, nthElem)
        weights[weightCords[0]][weightCords[1]][weightCords[2]] += change
        return weights[0], weights[1]

    def optimiseSingleWeight(self, loss, positiveDense1, positiveDense2, negativeDense1, negativeDense2, dense1, dense2, nthElem):
        positiveDense1.weights, positiveDense2.weights = self.changeWeight([copy.deepcopy(positiveDense1.weights), copy.deepcopy(positiveDense2.weights)], nthElem, 0.05)
        negativeDense1.weights, negativeDense2.weights = self.changeWeight([copy.deepcopy(negativeDense1.weights), copy.deepcopy(negativeDense2.weights)], nthElem, -0.05)

        positiveLoss = iteration(self.X, self.y, positiveDense1, positiveDense2, self.activation1, self.activation2)
        negativeLoss = iteration(self.X, self.y, negativeDense1, negativeDense2, self.activation1, self.activation2)

        losses = [(positiveLoss, positiveDense1, positiveDense2),
                  (negativeLoss, negativeDense1, negativeDense2),
                  (loss, dense1, dense2)]
        return min(losses, key=lambda x: x[0]) 

     
    def optimise(self, loss, dense1, dense2, nthElem):
        

        for nthElem in range(sum(len(inner) for outer in [copy.deepcopy(dense1.weights), copy.deepcopy(dense2.weights)] for inner in outer)):
            positiveDense1, positiveDense2 = copy.deepcopy(dense1), copy.deepcopy(dense2)
            negativeDense1, negativeDense2 = copy.deepcopy(dense1), copy.deepcopy(dense2)

            loss, dense1, dense2 = self.optimiseSingleWeight(loss, positiveDense1, positiveDense2, negativeDense1, negativeDense2, dense1, dense2, nthElem)

        # positiveDense1, positiveDense2 = copy.deepcopy(dense1), copy.deepcopy(dense2)
        # negativeDense1, negativeDense2 = copy.deepcopy(dense1), copy.deepcopy(dense2)

        # loss, dense1, dense2 = self.optimiseSingleWeight(loss, positiveDense1, positiveDense2, negativeDense1, negativeDense2, dense1, dense2, nthElem)

        return loss, dense1, dense2
