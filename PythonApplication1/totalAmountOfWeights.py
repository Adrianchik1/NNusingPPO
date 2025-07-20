def totalAmountOfWeights(denses, parameter):
    weights = []
    totalAmount = 0
    for dense in denses:
        weights.append(getattr(dense, parameter))
    for weight in weights:
        totalAmount += len(weight) * len(weight[0]) 
    return totalAmount