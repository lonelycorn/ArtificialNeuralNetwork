import argparse
import numpy as np
from sklearn import datasets
from sklearn import tree
from sklearn import ensemble
from sklearn import neighbors
from sklearn import model_selection


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



def normalize(raw):
    """
    :input raw: each row is a sample and each col is a feature channel
    :return normalized samples, per-channel avg, per-channel std
    """
    avg = np.mean(raw, axis=0)
    std = np.std(raw, axis=0)
    normalized = (raw - avg) / std
    return (normalized, avg, std)


def denormalize(normalized, avg, std):
    raw = normalized * std + avg
    return raw


def kernelize(raw):
    (sampleCount, N) = raw.shape

    # augmented features consist of
    #   - raw features (N)
    #   - second-order terms (N + 1) * N / 2
    #   - geometrical features (N - 1) * N / 2 + 1
    M1 = N
    M2 = int((N + 1) * N / 2)
    M3 = int((N - 1) * N / 2 + 1)
    M = M1 + M2 + M3

    output = np.zeros((sampleCount, M))

    output[:, 0 : M1] = raw

    k = M1
    for i in range(N):
        for j in range(i, N):
            output[:, k] = raw[:, i] * raw[:, j]
            k += 1

    for i in range(N):
        for j in range(i + 1, N):
            output[:, k] = np.arctan2(raw[:, i], raw[:, j])
            k += 1
    output[:, k] = np.linalg.norm(raw, axis=1)

    return output


def trainAndTest(X, Y, algorithm, useKernels, normalizeInput):

    testCount = 10
    bingo = np.zeros(testCount, dtype=np.int)

    for i in range(testCount):
        print("[%d/%d] Training using %s samples" % (i + 1, testCount, ["raw", "kernelized"][useKernels]))
        (trainingX, testingX, trainingY, testingY) = model_selection.train_test_split(X, Y, test_size=0.3)
        if (useKernels):
            trainingX = kernelize(trainingX)
            testingX = kernelize(testingX)

        if (normalizeInput):
            (trainingX, _, _) = normalize(trainingX)
            (testingX, _, _) = normalize(testingX)

        predictor = algorithm(trainingX, trainingY)
        predictions = predictor(testingX)

        bingo[i] = 0
        N = len(predictions)
        for (p, t) in zip(predictions, testingY):
            if (p == t):
                bingo[i] += 1
        accuracy = 1.0 * bingo[i] / N
        print(f"\t accuracy = {accuracy * 100 :.2f}% ({bingo[i]}/{N})")

    avg = np.sum(bingo) / (testCount * N) * 100
    std = np.std(bingo / N) * 100
    print(f"Accuracy avg = {avg:.2f}%, std = {std:.2f}%")


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

    trainAndTest(X, Y, algorithms[args.algorithm], args.useKernels, args.normalizeInput)

