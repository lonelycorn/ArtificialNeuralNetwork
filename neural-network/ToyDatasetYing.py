#!/usr/bin/env python3
import argparse
import logging
import numpy as np
import sys

from sklearn.datasets import (
    load_boston,
    load_iris,
    load_wine,
)
from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
)
from sklearn.neighbors import (
    KNeighborsClassifier,
    KNeighborsRegressor,
)
from sklearn.neural_network import (
    MLPClassifier,
    MLPRegressor,
)
from sklearn.model_selection import train_test_split
from sklearn.svm import (
    SVC,
    SVR,
)

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
LOGGER.addHandler(logging.StreamHandler(sys.stdout))


def trainBoston():
    rs = 42
    input, output = load_boston(return_X_y=True)
    LOGGER.debug(input.shape)

    trainInput, testInput, trainOutput, testOutput = train_test_split(
        input, output, test_size=0.2, random_state=rs
    )

    nn = MLPRegressor(
        solver='lbfgs', alpha=1e-5,
        hidden_layer_sizes=(6, 3), random_state=rs, batch_size=10,
        max_iter=1000
    ).fit(trainInput, trainOutput)
    nnOutput = nn.predict(testInput)
    LOGGER.info(f"NN: std: {np.std(nnOutput-testOutput)}")

    dt = DecisionTreeRegressor(random_state=rs).fit(trainInput, trainOutput)
    dtOutput = dt.predict(testInput)
    LOGGER.info(f"DT: std: {np.std(dtOutput-testOutput)}")

    knn = KNeighborsRegressor(n_neighbors=10).fit(trainInput, trainOutput)
    knnOutput = knn.predict(testInput)
    LOGGER.info(f"KNN: std: {np.std(knnOutput-testOutput)}")

    svm = SVR().fit(trainInput, trainOutput)
    svmOutput = svm.predict(testInput)
    LOGGER.info(f"SVM: std: {np.std(svmOutput-testOutput)}")

    LOGGER.info(f"avg: {np.std((svmOutput + knnOutput + nnOutput + dtOutput)/4 - testOutput)}")

    '''
    LOGGER.info(np.array_str(nnOutput, precision=2, max_line_width=100))
    LOGGER.info(np.array_str(dtOutput, precision=2, max_line_width=100))
    LOGGER.info(np.array_str(knnOutput, precision=2, max_line_width=100))
    LOGGER.info(np.array_str(svmOutput, precision=2, max_line_width=100))
    LOGGER.info(np.array_str((svmOutput + knnOutput + nnOutput + dtOutput)/4, precision=2, max_line_width=100))
    LOGGER.info(np.array_str(testOutput, precision=2, max_line_width=100))
    '''

def compareProbaToLabels(proba, labels, tag):
    '''
    compare predicted probability to test output lables, printing with tag
    will pring out the the accuracy and the probability in failure case
    @proba: predicted probability
    @lables: output labels in truth.
    @tag: priting tag
    '''
    predicts = np.argmax(proba, axis=1)
    size = len(proba)
    corrects = 0
    LOGGER.info(f"{tag}:")
    for index in range(size):
        #print(np.array_str(proba[index], precision=2))
        if predicts[index] == labels[index]:
            corrects += 1
        else:
            LOGGER.debug(f"{index}: label vs guess: {labels[index]} vs {predicts[index]}"
                         f" ,proba: {np.array_str(proba[index], precision=2)}")
    LOGGER.info(f"accuracy: {corrects/size}")

def trainIris():
    rs = 0
    input, output = load_iris(return_X_y=True)
    LOGGER.debug(input.shape)

    trainInput, testInput, trainOutput, testOutput = train_test_split(
        input, output, test_size=0.4, random_state=rs
    )

    '''
    trainInput = input
    testInput = input
    trainOutput = output
    testOutput = output
    '''

    nn = MLPClassifier(
        solver='lbfgs', alpha=1e-5,
        random_state=rs, batch_size=10,
        max_iter=1000
    ).fit(trainInput, trainOutput)
    nnOutput = nn.predict_proba(testInput)
    compareProbaToLabels(nnOutput, testOutput, "NN")

    dt = DecisionTreeClassifier(random_state=rs).fit(trainInput, trainOutput)
    dtOutput = dt.predict_proba(testInput)
    compareProbaToLabels(dtOutput, testOutput, "DT")

    knn = KNeighborsClassifier(n_neighbors=10).fit(trainInput, trainOutput)
    knnOutput = knn.predict_proba(testInput)
    compareProbaToLabels(knnOutput, testOutput, "KNN")

    svm = SVC(probability=True).fit(trainInput, trainOutput)
    svmOutput = svm.predict_proba(testInput)
    compareProbaToLabels(svmOutput, testOutput, "SVM")

    compareProbaToLabels((svmOutput + knnOutput + nnOutput + dtOutput)/4, testOutput, "avg")

def trainWine():
    rs = 0
    input, output = load_wine(return_X_y=True)
    LOGGER.debug(input.shape)

    trainInput, testInput, trainOutput, testOutput = train_test_split(
        input, output, test_size=0.4, random_state=rs
    )

    '''
    trainInput = input
    testInput = input
    trainOutput = output
    testOutput = output
    '''

    nn = MLPClassifier(
        solver='lbfgs', alpha=1e-5,
        random_state=rs, batch_size=10,
        max_iter=1000
    ).fit(trainInput, trainOutput)
    nnOutput = nn.predict_proba(testInput)
    compareProbaToLabels(nnOutput, testOutput, "NN")

    dt = DecisionTreeClassifier(random_state=rs).fit(trainInput, trainOutput)
    dtOutput = dt.predict_proba(testInput)
    compareProbaToLabels(dtOutput, testOutput, "DT")

    knn = KNeighborsClassifier(n_neighbors=10).fit(trainInput, trainOutput)
    knnOutput = knn.predict_proba(testInput)
    compareProbaToLabels(knnOutput, testOutput, "KNN")

    svm = SVC(probability=True).fit(trainInput, trainOutput)
    svmOutput = svm.predict_proba(testInput)
    compareProbaToLabels(svmOutput, testOutput, "SVM")

    compareProbaToLabels((svmOutput + knnOutput + nnOutput + dtOutput)/4, testOutput, "avg")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="use boston|iris|wine to load the dataset", type=str)
    args = parser.parse_args()
    if args.dataset == "boston":
        trainBoston()
    elif args.dataset == "iris":
        trainIris()
    elif args.dataset == "wine":
        trainWine()
    else:
        LOGGER.error("please input a dataset")
        exit()
