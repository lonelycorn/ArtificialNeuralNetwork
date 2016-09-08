import pickle # this is slower than cPickle
import numpy as np


def load_mnist_dataset():
    with open('./mnist.pkl', 'rb') as f:
        training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
        return _format_data(training_data), _format_data(validation_data), _format_data(test_data)

def _format_data(data):
    data_inputs     = [np.reshape(x, (784, 1)) for x in data[0]]
    data_outputs    = [np.zeros((10, 1)) for y in data[1]]
    for (i, y) in enumerate(data[1]):
        data_outputs[i][y][0] = 1.0

    return [(x, y) for (x, y) in zip(data_inputs, data_outputs)]

