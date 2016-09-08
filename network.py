import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '.'))

import itertools
import numpy as np
import activation_function
from layer import InputLayer, OutputLayer, Layer

class NetworkBaseClass:
    ID_NONE = -1
    _ID_generator = itertools.count(ID_NONE + 1)
    @staticmethod
    def GenerateID():
        """
        Generate a unique ID
        """
        return next(LayerBaseClass._ID_generator)

    def __init__(self):
        self._ID = NeutronBaseClass.GenerateID()

    @property
    def ID(self):
        return self._ID

    def Train(self, input, output, **training_params):
        raise NotImplementedError("Must implement Train().")

    def GetOutput(self, input):
        """
        Get the output of the network.
        === INPUT ===
        input:  a numpy ndarray; the input signal.
                NOTE: must conform to the sepcification of the input of
                      the network.
        """
        raise NotImplementedError("Must implement GetOutput().")

class FullyConnectedFeedforwardNetwork(NetworkBaseClass):
    def __init__(self, layer_sizes, af_type_str="sigmoid"):
        """
        === INPUT ===
        layer_sizes:    an numpy array; each element is an integer representing
                the number of neurons in that layer.
                NOTE: the first and the last elements define the number of input
                      and output signals, respectively.
        af_type_str:    a string; the type of the activation function to use.
                NOTE: see activation_function.Build() for details.
        """
        if (len(layer_sizes) < 2):
            raise RuntimeError("Must define at least 2 layers.")

        af = activation_function.Build(af_type_str)

        input_count = layer_sizes[0]
        self._input_layer = InputLayer(input_count)
        last_layer = self._input_layer
        for i in range(1, len(layer_sizes)):
            layer = Layer(layer_sizes[i], input_count, af)
            last_layer.SetNextLayer(layer)

            # roll forward
            input_count = layer_sizes[i]
            last_layer = layer

    def Train(self, batch, initial_rho):
        """
        === INPUT ===
        batch: a list of (x, y); the training input and output
        """
        for (x, y) in batch:
            #rho = initial_rho / (i + 1.0)
            rho = initial_rho
            self._input_layer.Train(x, y, rho)
        
        # debug: collect all biases
        #bias = []
        #layer = self._input_layer._next_layer
        #while (layer is not None):
        #    bias.append(layer._bias)
        #    layer = layer._next_layer
        #print("self._bias is \n{}".format(bias))
        #input("press any key to continue")

    def GetOutput(self, input):
        """
        === INPUT ===
        """
        raw_output = self._input_layer.Feedforward(input)

        return np.asarray(raw_output)

class SimpleFeedforwardNetwork(NetworkBaseClass):
    def __init__(self, layer_sizes):
        """
        layer_sizes: a list; number of neurons in each layer.
        """
        self._af = activation_function.Build("sigmoid")
        self._num_layers = len(layer_sizes)
        # note the first layer is input layer and doesn't need bias / weight
        self._bias = [np.random.normal(0.0, 1.0, (x, 1)) \
                for x in layer_sizes[1:]]
        self._weight = [np.random.normal(0.0, 1.0/np.sqrt(y), (x, y)) \
                for x, y in zip(layer_sizes[1:], layer_sizes[:-1])]

    def GetOutput(self, x):
        a = x # activation of the input layer is just the input
        for (b, w) in zip(self._bias, self._weight):
            z = b + np.dot(w, a)
            a = self._af.GetValue(z)
        return a

    def Train(self, batch, rho):
        """
        batch: a list of (x, y); the training input and output
        """
        # gradient of cost w.r.t. bias and weight
        nabla_b = [np.zeros(b.shape) for b in self._bias]   # partial C / partial b
        nabla_w = [np.zeros(w.shape) for w in self._weight] # partial C / partial w

        for (x, y) in batch:
            #print("y is \n{}".format(y))
            delta_nabla_b = [np.zeros(b.shape) for b in self._bias]
            delta_nabla_w = [np.zeros(w.shape) for w in self._weight]
            a = x
            all_a = [a]
            all_z = []
            ##### forward #####
            for (b, w) in zip(self._bias, self._weight):
                z = b + np.dot(w, a)
                a = self._af.GetValue(z)
                all_z.append(z)
                all_a.append(a)

            ##### at the end #####
            e = y - all_a[-1] # difference between expected y and last-layer activation
            # quadratic cost
            dC_over_dz = -e * self._af.GetDerivative(z[-1])
            # cross-entropy cost
            #dC_over_dz = -e
            delta_nabla_b[-1] = dC_over_dz
            delta_nabla_w[-1] = np.dot(dC_over_dz, all_a[-2].T)

            ##### backward #####
            for l in range(2, self._num_layers):
                dC_over_da = np.dot(self._weight[-l+1].T, dC_over_dz)
                dC_over_dz = dC_over_da * self._af.GetDerivative(all_z[-l])
                delta_nabla_b[-l] = dC_over_dz
                delta_nabla_w[-l] = np.dot(dC_over_dz, all_a[-l-1].T)
                #print("dC_over_da is \n{}".format(dC_over_da))
                #print("dC_over_dz is \n{}".format(dC_over_dz))
            #print("all_a is \n{}".format(all_a))
            #print("all_z is \n{}".format(all_z))
            #print("delta_nabla_w is \n{}".format(delta_nabla_w))
            #print("delta_nabla_b is \n{}".format(delta_nabla_b))

            nabla_b = [nb + dnb \
                    for (nb, dnb) in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw \
                    for (nw, dnw) in zip(nabla_w, delta_nabla_w)]



        # gradient descent
        n = len(batch)
        self._weight = [w - rho * dw / n \
                for (w, dw) in zip(self._weight, nabla_w)]

        self._bias = [b - rho * db / n \
                for (b, db) in zip(self._bias, nabla_b)]
        #print("nabla_w is \n{}".format(nabla_w))
        #print("nabla_b is \n{}".format(nabla_b))
        #print("self._weight is \n{}".format(self._weight))
        #print("self._bias is \n{}".format(self._bias))
        #input("press any key to continue")


if (__name__ == "__main__"):
    pass
