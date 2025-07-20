import numpy as np
import copy
from iterate import iteration
from totalAmountOfWeights import totalAmountOfWeights

class Optimiser():
    def __init__(self, X, y, activation1, activation2, change):
        self.X = X
        self.y = y
        self.activation1 = activation1
        self.activation2 = activation2
        self.change = change

    def getParameterCords(self, WeightsOrBiases, nthElem):
        for x,m in enumerate(WeightsOrBiases):
            r,c=len(m),len(m[0])
            if nthElem<r*c:return[x,nthElem//c,nthElem%c]
            nthElem-=r*c
    def changeWeight(self, weights, nthElem, change):
        weightCords = self.getParameterCords(weights, nthElem)
        weights[weightCords[0]][weightCords[1]][weightCords[2]] += change
        return weights

    def optimiseSingleWeight(self, loss, positiveDense1 ,positiveDense2, negativeDense1, negativeDense2, dense1, dense2, nthElem):
        positiveDense1.weights, positiveDense2.weights = self.changeWeight([copy.deepcopy(positiveDense1.weights), copy.deepcopy(positiveDense2.weights)], nthElem, self.change)
        negativeDense1.weights, negativeDense2.weights = self.changeWeight([copy.deepcopy(negativeDense1.weights), copy.deepcopy(negativeDense2.weights)], nthElem, -self.change)

        positiveLoss = iteration(self.X, self.y, positiveDense1, positiveDense2, self.activation1, self.activation2)
        negativeLoss = iteration(self.X, self.y, negativeDense1, negativeDense2, self.activation1, self.activation2)

        losses = [(positiveLoss, positiveDense1, positiveDense2),
                  (negativeLoss, negativeDense1, negativeDense2),
                  (loss, dense1, dense2)] 
        return min(losses, key=lambda x: x[0]) 

    def optimiseSingleBias(self, loss, positiveDense1 ,positiveDense2, negativeDense1, negativeDense2, dense1, dense2, nthElem):
        positiveDense1.biases, positiveDense2.biases = self.changeWeight([copy.deepcopy(positiveDense1.biases), copy.deepcopy(positiveDense2.biases)], nthElem, self.change)
        negativeDense1.biases, negativeDense2.biases = self.changeWeight([copy.deepcopy(negativeDense1.biases), copy.deepcopy(negativeDense2.biases)], nthElem, -self.change)

        positiveLoss = iteration(self.X, self.y, positiveDense1, positiveDense2, self.activation1, self.activation2)
        negativeLoss = iteration(self.X, self.y, negativeDense1, negativeDense2, self.activation1, self.activation2)

        losses = [(positiveLoss, positiveDense1, positiveDense2),
                  (negativeLoss, negativeDense1, negativeDense2),
                  (loss, dense1, dense2)] 
        return min(losses, key=lambda x: x[0]) 
     
     
    def optimise(self, loss, denses): 
        dense1, dense2 = denses[0], denses[1]
        for nthElem in range(totalAmountOfWeights(denses, "weights")):
            positiveDense1, positiveDense2 = copy.deepcopy(dense1), copy.deepcopy(dense2)
            negativeDense1, negativeDense2 = copy.deepcopy(dense1), copy.deepcopy(dense2)

            loss, dense1, dense2 = self.optimiseSingleWeight(loss, positiveDense1 ,positiveDense2, negativeDense1, negativeDense2, dense1, dense2, nthElem)
            if nthElem%len(dense1.biases[0]) == 0: nthElem = int(nthElem/len(dense1.biases[0])); loss, dense1, dense2 = self.optimiseSingleBias(loss, positiveDense1 ,positiveDense2, negativeDense1, negativeDense2, dense1, dense2, nthElem)

        # positiveDense1, positiveDense2 = copy.deepcopy(dense1), copy.deepcopy(dense2)
        # negativeDense1, negativeDense2 = copy.deepcopy(dense1), copy.deepcopy(dense2)

        # loss, dense1, dense2 = self.optimiseSingleWeight(loss, positiveDense1, positiveDense2, negativeDense1, negativeDense2, dense1, dense2, nthElem)

        return loss, dense1, dense2

