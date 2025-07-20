from losses import Loss_CategoricalCrossentrophy

def iteration(X, y, denses, activations):
    # dense1.forward(X)
    # activation1.forward(dense1.output)

    # dense2.forward(activation1.output)
    # activation2.forward(dense2.output)
    for dense, activation in zip(denses, activations):
        dense.forward(X)
        activation.forward(dense.output)
        X = activation.output

    loss_function = Loss_CategoricalCrossentrophy()
    loss = loss_function.calculate(X, y)

    return loss