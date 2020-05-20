import argparse
import numpy as np
from sklearn import datasets
from sklearn import tree
from sklearn import ensemble
from sklearn import neighbors

import utils


def trainDecisionTree(X, Y):
    """
    :return a decision tree predictor
    """
    regressor = tree.DecisionTreeRegressor()
    regressor.fit(X, Y)

    def predict(X):
        return regressor.predict(X)

    return predict


def trainRandomForest(X, Y):
    """
    :return a random forest predictor
    """
    regressor = ensemble.RandomForestRegressor(max_samples=0.75)
    regressor.fit(X, Y)

    def predict(X):
        return regressor.predict(X)

    return predict


def trainNeuralNet(X, Y):
    """
    :return a neural net learner
    """
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.utils import to_categorical

    model = Sequential()
    # NOTE: input "layer" is represented by the "input shape"
    # first layer: 30 hidden neurons
    model.add(Dense(
        input_shape=(X.shape[1], ),
        units=30,
        activation="relu",
        use_bias=True,
        kernel_initializer="he_normal",
        bias_initializer="zeros"
    ))
    # second layer: 10 hidden neurons
    model.add(Dense(
        units=10,
        activation="relu",
        use_bias=True,
        kernel_initializer="he_normal",
        bias_initializer="zeros"
    ))
    # last layer: 1 output, the predicted median
    model.add(Dense(
        units=1,
        activation="relu",
        use_bias=True,
        kernel_initializer="he_normal",
        bias_initializer="zeros"
    ))

    model.compile(optimizer="adadelta",loss="mean_squared_error", metrics=["accuracy"])

    model.fit(X, Y, batch_size=10, epochs=100, verbose=0)

    def predict(X):
        return model.predict(X)

    return predict


def trainKnn(X, Y):
    """
    :return a KNN predictor
    """
    regressor = neighbors.KNeighborsRegressor(n_neighbors=5, weights="distance")
    regressor.fit(X, Y)

    def predict(X):
        return regressor.predict(X)

    return predict


if (__name__ == "__main__"):

    algorithms = {
            "decision-tree": trainDecisionTree,
            "random-forest": trainRandomForest,
            "neural-net": trainNeuralNet,
            "knn": trainKnn,
    }

    parser = argparse.ArgumentParser(description="supervised learning trains on the BOSTON dataset")
    parser.add_argument("algorithm", choices=algorithms.keys())
    parser.add_argument("--use-kernels", dest="useKernels", action="store_true")
    parser.add_argument("--normalize-input", dest="normalizeInput", action="store_true")

    args = parser.parse_args()

    print("Loading BOSTON dataset")
    boston = datasets.load_boston()
    #  506x13 array, row-major
    X = boston.data
    Y = boston.target

    utils.trainAndTestRegressor(X, Y, algorithms[args.algorithm], args.useKernels, args.normalizeInput)

