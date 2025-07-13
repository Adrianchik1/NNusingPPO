from losses import Loss_CategoricalCrossentrophy

def iteration(X, y, dense1, dense2, activation1, activation2):
    dense1.forward(X)
    activation1.forward(dense1.output)

    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    loss_function = Loss_CategoricalCrossentrophy()
    loss = loss_function.calculate(activation2.output, y)

    return loss