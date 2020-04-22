import numpy as np

def create(name):
    cfs = [Quadratic, CrossEntropy]
    for cf in cfs:
        if (cf.getName() == name):
            return cf()
    raise ValueError(f"Unsupported cost function: '{name}'")

class Interface:
    def __init__(self):
        pass

    def getValue(self, y, yHat):
        raise NotImplementedError("TODO")

    def getValue(self, y, yHat):
        raise NotImplementedError("TODO")

    def getValueAndDerivative(self, y, yHat):
        return (self.getValue(y, yHat), self.getDerivative(y, yHat))


class Quadratic(Interface):
    @staticmethod
    def getName():
        return "quadratic"

    def __init__(self):
        pass

    def getValue(self, y, yHat):
        y = np.asarray(y)
        yHat = np.asarray(yHat)
        e = yHat - y
        return 0.5 * np.dot(e, e)

    def getDerivative(self, y, yHat):
        y = np.asarray(y)
        yHat = np.asarray(yHat)
        e = yHat - y
        return e

    def getValueAndDerivative(self, y, yHat):
        y = np.asarray(y)
        yHat = np.asarray(yHat)
        e = yHat - y
        return (0.5 * np.dot(e, e), e)

class CrossEntropy:
    @staticmethod
    def getName():
        return "cross_entropy"

    def __init__(self):
        pass

    def getValue(self, y, yHat):
        return -(y * np.log(yHat) + (1.0 - y) * np.log(1.0 - yHat))

    def getDerivative(self, y, yHat):
        return -(y / yHat - (1.0 - y) / (1.0 - yHat))

    def getValueAndDerivative(self, y, yHat):
        a = 1.0 - y
        b = 1.0 - yHat
        return (-(y * np.log(yHat) + a * np.log(b)), -(y / yHat - a / b))

if (__name__ == "__main__"):
    from matplotlib import pyplot as plt

    def doPlot(name):
        cf = create(name)
        fig = plt.figure()

        y = 0.5
        yHat = np.linspace(-0.4, 0.4, 100) + y
        C = [cf.getValue(y, v) for v in yHat]

        plt.plot(yHat, C)

        plt.legend(["C(yHat)", "C'(yHat)"])
        plt.title(f"{name} (y = 0.5)")

    doPlot(Quadratic.getName())
    doPlot(CrossEntropy.getName())

    plt.show()

