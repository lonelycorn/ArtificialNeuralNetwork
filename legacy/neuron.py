import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '.'))

import numpy as np
import itertools

class NeuronBaseClass:
    ID_NONE = -1
    _ID_generator = itertools.count(ID_NONE + 1)
    @staticmethod
    def GenerateID():
        """
        Generate a unique ID
        """
        return next(NeuronBaseClass._ID_generator)

    def __init__(self):
        self._ID = NeuronBaseClass.GenerateID()

    def __str__(self):
        raise NotImplementedError("Must implement __str__().")

    @property
    def ID(self):
        return self._ID

    def SetNextNeuron(self, next_neutron):
        raise NotImplementedError("Must implement SetNextNeuron().")

    def SetPreviousNeuron(self, prev_neutron):
        raise NotImplementedError("Must implement SetPreviousNeuron().")

    def GetOutput(self):
        raise NotImplementedError("Must implement GetValue().")

class StandaloneNeuron(NeuronBaseClass):
    def __init__(self, number_of_inputs, activation_function, learning_coeff, supervised=True):
        NeuronBaseClass.__init__(self)
        self._activation_function = activation_function
        self._learning_coeff = learning_coeff * 1.0
        if (supervised):
            self._do_train = self._do_train_supervised
        else:
            self._do_train = self._do_train_unsupervised
        self._weight = np.ones(number_of_inputs) # inputs are 1, x_1, x_2, ..., x_n
        self._bias = 1.0 # weight for the constant

    def _get_input_sum(self, input):
        return sum(input * self._weight) + self._bias

    def _do_train_supervised(self, input, expected_output):
        """
        Supervised learning.
        Train the weights of the Neuron by minimizing the square error
        of the output.
        === INPUT ===
        input: numpy array [x_1, x_2, ..., x_n]
        expected_output: a scalar value.
        """
        # to minimize e^2, we use gradient descent:
        #   dw_k = - rho * d(e^2) / d(x_k) = - 2 * rho * e * d(-af(input_sum))/d(input_sum) * x_k
        for iteration in range(1, 500):
            rho = self._learning_coeff / iteration
            input_sum = self._get_input_sum(input)
            output = self.GetOutput(input)
            error = expected_output - output
            af_prime = self._activation_function.GetDerivative(input_sum)
            self._weight += 2 * rho * error * af_prime * input
            self._bias   += 2 * rho * error * af_prime * 1.0 

    def _do_train_unsupervised(self, input):
        """
        Unsupervised learning.
        Train the weights of the Neuron by minimizing the square magnitude
        of the output.
        === INPUT ===
        input: numpy array [x_1, x_2, ..., x_n]
        """
        # to minimize output^2, we use gradient descent:
        #   dw_k = - rho * d(output^2) / d(w_k) = - 2 * rho * output * d(-af(input_sum))/d(input_sum) * x_k
        input_sum = self._get_input_sum(input)
        output = self.GetOutput(input)
        af_prime = self._activation_function.GetDerivative(input_sum)
        self._weight += 2 * self._rho * output * af_prime * input
        self._bias   += 2 * self._rho * output * af_prime * 1.0 

    def _do_train(self, *args):
        pass

    def Train(self, *args):
        """
        Train the neuron
        === INPUT ===
        === OUTPUT ===
        the error after training?
        """
        self._do_train(*args)

    def GetOutput(self, input):
        """
        Get the output of the neutron with current weights.
        === INPUT ===
        input: numpy array [x_1, x_2, ..., x_n]
        === OUTPUT ===
        a scalar, the output of the neutron.
        """
        return self._activation_function.GetValue(self._get_input_sum(input)) 

class Neuron(NeuronBaseClass):
    def __init__(self, activation_function):
        NeuronBaseClass.__init__()
        self._activation_function = activation_function


class InputNeuron(NeuronBaseClass):
    def __init__(self):
        NeuronBaseClass.__init__()


class OutputNeuron(NeuronBaseClass):
    def __init__(self):
        NeuronBaseClass.__init__()


        
