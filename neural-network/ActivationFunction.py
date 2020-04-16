import numpy as np

class Sigmoid:
    def __init__(self):
        pass

    def getValue(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def getDerivative(self, x):
        y = self.getValue(x)
        return (1.0 - y) * y

class Tanh:
    def __init__(self):
        pass

    def getValue(self, x):
        return np.tanh(x)

    def getDerivative(self, x):
        y = self.getValue(x)
        return 1.0 - y * y

class Relu:
    """
    A piecewise linear function that is fast to compute, and fewer vanishing
    gradient problems. The only issue is dead neurons: with random initial weight
    and fast learning rate, some units may never activate. To mitigate that, a
    new parameter leakRate is introduced, which is slope when x < 0
    """
    def __init__(self, leakRate=0.01):
        self.mLeakRate = leakRate

    def getValue(self, x):
        return np.where(x > 0, x, x * self.mLeakRate)

    def getDerivative(self, x):
        v = np.ones(len(x))
        return np.where(x > 0, v, v * self.mLeakRate)

class Softplus:
    def __init__(self):
        pass

    def getValue(self, x):
        return np.log(1.0 + np.exp(x))

    def getDerivative(self, x):
        return 1.0 / (1.0 + np.exp(-x))

class Linear:
    """
    Using this function in a neural net turns it into a standard linear regressor.
    This is introduced for InputNeuron to pass-through the input
    """
    def __init__(self):
        pass

    def getValue(self, x):
        return x

    def getDerivative(self, x):
        return np.ones(len(x))

if (__name__ == "__main__"):
    from matplotlib import pyplot as plt

    def doPlot(af, title):
        fig = plt.figure()

        x = np.linspace(-2.5, 2.5, 100)
        #y = np.array([af.getValue(i) for i in x])
        #dy_over_dx = np.array([af.getDerivative(i) for i in x])
        y = af.getValue(x)
        dy_over_dx = af.getDerivative(x)

        plt.plot(x, y)
        plt.plot(x, dy_over_dx)

        plt.legend(["f(x)", "f'(x)"])
        plt.title(title)

    doPlot(Sigmoid(), "sigmoid")
    doPlot(Tanh(), "tanh")
    doPlot(Relu(), "relu")
    doPlot(Softplus(), "softplus")
    doPlot(Linear(), "linear")

    plt.show()
