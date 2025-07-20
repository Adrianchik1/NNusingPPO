import numpy as np
import copy
from iterate import iteration
from weightsBiases import Layer_Dense

class Optimiser():
    def __init__(self, X, y, activations, change):
        self.X = X
        self.y = y
        self.activations = activations
        self.change = change

    def optimiseSingleWeight(self, loss, denses, nthElem):
        
        positiveDenses = Layer_Dense.ChangeWeight(denses, nthElem, self.change)
        negativeDenses = Layer_Dense.ChangeWeight(denses, nthElem, -self.change)

        positiveLoss = iteration(self.X, self.y, positiveDenses, self.activations)
        negativeLoss = iteration(self.X, self.y, negativeDenses, self.activations)
        loss = iteration(self.X, self.y, denses, self.activations)

        losses = [(positiveLoss, positiveDenses),
                  (negativeLoss, negativeDenses),
                  (loss, denses)] 
        return min(losses, key=lambda x: x[0]) 

    def optimiseSingleBias(self, loss, denses, nthElem):
        positiveDenses = Layer_Dense.ChangeBias(denses, nthElem, self.change)
        negativeDenses = Layer_Dense.ChangeBias(denses, nthElem, -self.change)

        positiveLoss = iteration(self.X, self.y, positiveDenses, self.activations)
        negativeLoss = iteration(self.X, self.y, negativeDenses, self.activations)

        losses = [(positiveLoss, positiveDenses),
                  (negativeLoss, negativeDenses),
                  (loss, denses)] 
        return min(losses, key=lambda x: x[0]) 
     
     
    def optimise(self, loss, denses):
        for whichWeight in range(Layer_Dense.totalAmountOfWeights):
            loss, denses = self.optimiseSingleWeight(loss, denses, whichWeight)
            
            if whichWeight%len(denses[0].biases[0]) == 0: whichWeight = int(whichWeight/len(denses[0].biases[0])); loss, denses = self.optimiseSingleBias(loss, denses, whichWeight)

        return loss, denses

