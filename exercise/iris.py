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
    classifier = tree.DecisionTreeClassifier()
    classifier.fit(X, Y)

    def predict(X):
        return classifier.predict(X)

    return predict


def trainRandomForest(X, Y):
    """
    :return a random forest predictor
    """
    classifier = ensemble.RandomForestClassifier(max_samples=0.75)
    classifier.fit(X, Y)

    def predict(X):
        return classifier.predict(X)

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
    # first layer: 10 hidden neurons
    model.add(Dense(
        input_shape=(X.shape[1], ),
        units=10,
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
    # last layer: 3 output probabilities
    model.add(Dense(
        units=3,
        activation="softmax",
        use_bias=True,
        kernel_initializer="he_normal",
        bias_initializer="zeros"
    ))

    model.compile(optimizer="adadelta",loss="categorical_crossentropy", metrics=["accuracy"])

    Y = to_categorical(Y, 3)
    model.fit(X, Y, batch_size=10, epochs=100, verbose=0)

    def predict(X):
        probabilities = model.predict(X)
        predictions = np.zeros(len(X))

        for (i, p) in enumerate(probabilities):
            predictions[i] = np.argmax(p)
        return predictions

    return predict


def trainKnn(X, Y):
    """
    :return a KNN predictor
    """
    classifier = neighbors.KNeighborsClassifier(n_neighbors=5, weights="distance")
    classifier.fit(X, Y)

    def predict(X):
        return classifier.predict(X)

    return predict


if (__name__ == "__main__"):

    algorithms = {
            "decision-tree": trainDecisionTree,
            "random-forest": trainRandomForest,
            "neural-net": trainNeuralNet,
            "knn": trainKnn,
    }

    parser = argparse.ArgumentParser(description="supervised learning trains on the IRIS dataset")
    parser.add_argument("algorithm", choices=algorithms.keys())
    parser.add_argument("--use-kernels", dest="useKernels", action="store_true")
    parser.add_argument("--normalize-input", dest="normalizeInput", action="store_true")

    args = parser.parse_args()

    print("Loading IRIS dataset")
    iris = datasets.load_iris()
    #  150x4 array, row-major.
    # (sepal length, sepal width, petal length, petal width)
    X = iris.data
    Y = iris.target

    utils.trainAndTestClassifier(X, Y, algorithms[args.algorithm], args.useKernels, args.normalizeInput)

