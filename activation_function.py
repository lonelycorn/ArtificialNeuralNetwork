import numpy as np

def Build(type_str="sigmoid"):
    """
    Factory method that builds the specified activation function.
    """
    if (type_str == "sigmoid"):
        return SigmoidAF()
    elif (type_str == "arctan"):
        return ArctanAF()
    else:
        raise NotImplementedError("Unsupported activation function: '%s'" % (type_str))

class ActivationFunctionBaseClass:
    """
    Base class that defines the interfaces of all activation functions.
    """
    def __init__(self):
        pass

    def GetValue(self, x):
        raise NotImplementedError("Must implement GetValue().")

    def GetDerivative(self, x):
        raise NotImplementedError("Must implement GetDerivative().")

class SigmoidAF(ActivationFunctionBaseClass):
    def __init__(self):
        pass

    def GetValue(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def GetDerivative(self, x):
        v = self.GetValue(x)
        return v * (1.0 - v)

class ArctanAF(ActivationFunctionBaseClass):
    def __init__(self):
        pass

    def GetValue(self, x):
        return np.arctan(x);

    def GetDerivative(self, x):
        return 1.0 / (1.0 + x * x)
