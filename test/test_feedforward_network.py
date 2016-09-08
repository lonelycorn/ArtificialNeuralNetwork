import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import time

import data_loader
from network import FullyConnectedFeedforwardNetwork
from network import SimpleFeedforwardNetwork
import numpy as np
import pickle


def compare_result(predicted, expected):
    predicted_idx = np.argmax(predicted)
    expected_idx = np.argmax(expected)
    #print("validating testcase: expected %d, got %d" % (expected_idx, predicted_idx))
    return predicted_idx == expected_idx

def validate_network(network, validation_data, max_data_count=None):
    """
    max_data_count
    return the failure rate
    """
    failure_count = 0
    total_count = 0
    for (vi, vo) in validation_data:
        total_count += 1
        predicted_output = network.GetOutput(vi)
        if (not compare_result(predicted_output, vo)):
            failure_count += 1
        if ((max_data_count is not None) and (total_count > max_data_count)):
            break

    result = failure_count / total_count
    return result



if (__name__ == "__main__"):
    layer_sizes = [28*28, 30, 10]
    rho = 0.5 
    batch_size = 10
    max_epoch = 10

    use_simple_network = False
    if (len(sys.argv) > 2):
        raise RuntimeError("Do not know how to handle the command-line arguments:\n{}".format(sys.argv))
    elif (len(sys.argv) > 1):
        use_simple_network = (sys.argv[1] == "simple")

    if (use_simple_network):
        print("testing simple feedforward network")
        n = SimpleFeedforwardNetwork(layer_sizes)
    else:
        print("testing fully connected feedforward network")
        n = FullyConnectedFeedforwardNetwork(layer_sizes, "sigmoid")

    # load dataset
    training_data, validation_data, test_data = \
            data_loader.load_mnist_dataset()

    # training
    for iter in range(max_epoch):
        start_time = time.time()
        actual_rho = rho / (iter + 1)
        np.random.shuffle(training_data)
        for k in range(batch_size, len(training_data), batch_size):
            #if (k > 10000):
            #    break
            batch = training_data[k-batch_size:k]
            n.Train(batch, rho)
        # run validation data
        error_rate = validate_network(n, validation_data, 100)
        end_time = time.time()
        print("epoch %d: %.2fs, error rate %.2f%%" % (iter, end_time - start_time, 100 * error_rate))
    pickle.dump(n, open("network.pkl", "wb"))

