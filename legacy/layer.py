import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '.'))

import itertools
import numpy as np

class LayerBaseClass:
    ID_NONE = -1
    _ID_generator = itertools.count(ID_NONE + 1)
    @staticmethod
    def GenerateID():
        """
        Generate a unique ID
        """
        return next(LayerBaseClass._ID_generator)

    def __init__(self):
        self._ID = LayerBaseClass.GenerateID()

    @property
    def ID(self):
        return self._ID

    @property
    def neurons(self):
        """
        Get the neurons in this layer.
        """
        raise NotImplementedError("Must implement neurons.")

    def GetOutput(self, input):
        """
        Get the output of this layer.
        === INPUT ===
        input:  a numpy ndarray; the input signal to this layer.
                NOTE: must be normalized; must conform to the specification
                      of the input to the layer.
        === OUTPUT ===
        output: a numpy ndarray; the output signal of this layer, as defined by
                the specification.
        """
        raise NotImplementedError("Must implement GetOutput().")

    def Train(self, input, output):
        """
        Train the layer with the given input and output.
        === INPUT ===
        === OUTPUT ===
        dC_over_da:    a numpy ndarray; the gradient of the cost w.r.t each input
        """
        raise NotImplementedError("Must implement Train().")

class Layer(LayerBaseClass):
    def __init__(self, neuron_count, input_count, activation_function):
        """
        Constructor.
        === INPUT ===
        neuron_count:   an integer; the number of neurons in this layer.
        input_count:    an integer; the number of input signals to this layer.
                NOTE: the input signals are connected to all neurons. 
        
        """
        LayerBaseClass.__init__(self)
        self._input_count = input_count
        weight_stddev = 1.0 / np.sqrt(input_count) # the initial weight is more centered
        self._bias = np.random.normal(0.0, 1.0, (neuron_count, 1))
        self._weight = np.random.normal(0.0, weight_stddev, (neuron_count, input_count))
        self._activation_function = activation_function
        self._next_layer = None

    @property
    def neurons(self):
        """
        For efficiency and performance, we don't store neurons explicitly;
        they are constructed only when asked for.
        === OUTPUT ===
        A list of neurons.
        """
        # TODO: implement this
        raise NotImplementedError("need to generate the neurons")
        neurons = []
        return neurons

    def _get_weighted_input(self, input):
        return self._bias + np.dot(self._weight, input)

    def GetOutput(self, input):
        #assert(input.shape == (self._input_count, 1))
        z = self._get_weighted_input(input)
        return self._activation_function.GetValue(z)

    def Feedforward(self, input):
        #assert(input.shape == (self._input_count, 1))
        a = self.GetOutput(input)
        if (self._next_layer is not None):
            output = self._next_layer.Feedforward(a)
        else: # this is the last layer
            output = a

        return output

    def Train(self, x, y, rho):
        #assert(x.shape == (self._input_count, 1))

        z = self._get_weighted_input(x)     # neuron_count * 1
        a = self._activation_function.GetValue(z)   # neuron_count * 1

        if (self._next_layer is not None):
            dC_over_da = self._next_layer.Train(a, y, rho) # neuron_count * 1
            dC_over_dz = dC_over_da * \
                    self._activation_function.GetDerivative(z)  # neuron_count * 1
            #print("dC_over_da is \n{}".format(dC_over_da))
            #print("dC_over_dz is \n{}".format(dC_over_dz))

        else: # this is the output layer
            e = y - a   # neuron_count * 1

            ## Quadratic cost
            dC_over_dz = -e * self._activation_function.GetDerivative(z)  # neuron_count * 1

            ## cross-entropy cost
            #dC_over_dz = -e # neuron_count * 1

            #print("--------")
            #print("y = \n{}".format(y))
            #print("e = \n{}".format(e))
            #print("magnitude = {}".format(sum(abs(e))))
            #input("Press any key to continue...")


        # gradient w.r.t to weight and bias
        nabla_b = dC_over_dz # neuron_count * 1
        nabla_w = np.dot(dC_over_dz, x.T) # neuron_count * input_count

        # compute the return value before we updated the weight
        dC_over_da_return = np.dot(self._weight.T, dC_over_dz) # input_count * 1

        # gradient descent
        self._bias   -= rho * nabla_b # neuron_count * 1
        self._weight -= rho * nabla_w # neuron_count * input_count

        #print("z = \n{}".format(z))
        #print("a = \n{}".format(a))
        #print("dC_over_dz = \n{}".format(dC_over_dz))
        #print("dC_over_da = \n{}".format(dC_over_da))
        #print("delta_b.T = {}".format(delta_b.T))
        #input("Press any key to continue...")
        return dC_over_da_return

    def SetNextLayer(self, next_layer):
        """
        Connect to the next layer.
        === INPUT ===
        next_layer:     a descendant of LayerBaseClass; the next layer.
        """
        self._next_layer = next_layer

class InputLayer(LayerBaseClass):
    def __init__(self, input_count):
        LayerBaseClass.__init__(self)
        self._next_layer = None
        self._input_count = input_count

    @property
    def neurons(self):
        raise RuntimeError("An input layer doesn't have neurons.")

    def GetOutput(self, input):
        assert(len(input) == self._input_count)
        return input;

    def Feedforward(self, input):
        assert(len(input) == self._input_count)
        assert(self._next_layer is not None)
        output = self._next_layer.Feedforward(input)
        return output

    def Train(self, input, output, rho):
        assert(len(input) == self._input_count)
        assert(self._next_layer is not None)
        self._next_layer.Train(input, output, rho)

    def SetNextLayer(self, next_layer):
        self._next_layer = next_layer

class OutputLayer(Layer):
    """
    We don't have to use this...
    """
    def __init__(self, neuron_count, input_count, activation_function):
        Layer.__init__(self, neuron_count, input_count, activation_function)

    def SetNextLayer(self, next_layer):
        raise RuntimeError("An output layer is the last layer.")

if (__name__ == "__main__"):
    pass
