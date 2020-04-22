import numpy as np

def create(name):
    afs = [Sigmoid, Tanh, Relu, Softplus, Linear]
    for af in afs:
        if (af.getName() == name):
            return af()
    raise InputError(f"Unsupported activation function: '{name}'")


class Interface:
    def __init__(self):
        pass

    def getValue(self, x):
        raise NotImplementedError("TODO")

    def getDerivative(self, x):
        raise NotImplementedError("TODO")

    def getValueAndDerivative(self, x):
        return (self.getValue(x), self.getDerivative(x))


class Sigmoid(Interface):
    @staticmethod
    def getName():
        return "sigmoid"

    def __init__(self):
        pass

    def getValue(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def getDerivative(self, x):
        y = self.getValue(x)
        return (1.0 - y) * y

    def getValueAndDerivative(self, x):
        y = self.getValue(x)
        return (y, (1.0 - y) * y)


class Tanh(Interface):
    @staticmethod
    def getName():
        return "tanh"

    def __init__(self):
        pass

    def getValue(self, x):
        return np.tanh(x)

    def getDerivative(self, x):
        y = self.getValue(x)
        return 1.0 - y * y

    def getValueAndDerivative(self, x):
        y = self.getValue(x)
        return (y, 1.0 - y * y)


class Relu(Interface):
    """
    A piecewise linear function that is fast to compute, and fewer vanishing
    gradient problems. The only issue is dead neurons: with random initial weight
    and fast learning rate, some units may never activate. To mitigate that, a
    new parameter leakRate is introduced, which is slope when x < 0
    """
    @staticmethod
    def getName():
        return "relu"

    def __init__(self, leakRate=0.01):
        self.mLeakRate = leakRate

    def getValue(self, x):
        return np.where(x > 0, x, x * self.mLeakRate)

    def getDerivative(self, x):
        return np.where(x > 0, 1.0, self.mLeakRate)


class Softplus(Interface):
    @staticmethod
    def getName():
        return "softplus"

    def __init__(self):
        pass

    def getValue(self, x):
        return np.log(1.0 + np.exp(x))

    def getDerivative(self, x):
        return 1.0 / (1.0 + np.exp(-x))


class Linear(Interface):
    """
    Using this function in a neural net turns it into a standard linear regressor.
    This is introduced for InputNeuron to pass-through the input
    """
    @staticmethod
    def getName():
        return "linear"

    def __init__(self):
        pass

    def getValue(self, x):
        return x

    def getDerivative(self, x):
        x = np.asarray(x)
        if (0 == x.ndim):
            return 1.0
        else:
            return np.ones(len(x))


if (__name__ == "__main__"):
    from matplotlib import pyplot as plt

    def doPlot(name):
        af = create(name)
        fig = plt.figure()

        x = np.linspace(-2.5, 2.5, 100)
        (y, dy_over_dx) = af.getValueAndDerivative(x)

        plt.plot(x, y)
        plt.plot(x, dy_over_dx)

        plt.legend(["f(x)", "f'(x)"])
        plt.title(name)

    doPlot(Sigmoid.getName())
    doPlot(Tanh.getName())
    doPlot(Relu.getName())
    doPlot(Softplus.getName())
    doPlot(Linear.getName())

    plt.show()
