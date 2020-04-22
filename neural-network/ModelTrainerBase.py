#!/usr/bin/env python3
import argparse
import json
import logging
import numpy
import sys

from MachineLearningUtils import (
    Constants,
    JsonError,
    Utils,
)

class ModelTrainerBase():
    def __init__(self, dataset: str, config: str):
        self.initDataset(dataset)
        self.parseConfig(config)

    def initDataset(self, dataset: str):
        if dataset == Constants.DATASET_MNIST:
            self.mTrainInput = numpy.fromfile(Constants.FILE_TRAIN_INPUT, dtype=numpy.uint8)
            self.mTrainOutput = numpy.fromfile(Constants.FILE_TRAIN_LABELS, dtype=numpy.uint8)
            self.mTestInput = numpy.fromfile(Constants.FILE_TEST_INPUT, dtype=numpy.uint8)
            self.mTestOutput = numpy.fromfile(Constants.FILE_TEST_LABELS, dtype=numpy.uint8)

            self.mNumTrainSamples = Utils.bytesToInt(self.mTrainInput, 0)
            self.mDimInput = Utils.bytesToInt(self.mTrainInput, 4)
            assert(self.mNumTrainSamples == Utils.bytesToInt(self.mTrainOutput, 0))
            assert(self.mDimInput == Utils.bytesToInt(self.mTestInput, 4))
            self.mNumTestSamples = Utils.bytesToInt(self.mTestInput, 0)
            assert(self.mNumTestSamples == Utils.bytesToInt(self.mTestOutput, 0))
            self.mDimOutput = 10

            self.mTrainInput = self.mTrainInput[8:].reshape(self.mNumTrainSamples, self.mDimInput).transpose()
            self.mTrainInput = self.mTrainInput / 255.0
            self.mTestInput = self.mTestInput[8:].reshape(self.mNumTestSamples, self.mDimInput).transpose()
            self.mTestInput = self.mTestInput / 255.0

            data = self.mTrainOutput[4:]
            self.mTrainOutput = numpy.zeros((self.mDimOutput, self.mNumTrainSamples))
            for index in range(0, self.mNumTrainSamples):
                self.mTrainOutput[data[index], index] = 1

            data = self.mTestOutput[4:]
            self.mTestOutput = numpy.zeros((self.mDimOutput, self.mNumTestSamples))
            for index in range(0, self.mNumTestSamples):
                self.mTestOutput[data[index], index] = 1

        else:
            raise ValueError(f"Could not init Dataset {dataset}")

    def parseConfig(self, config: str):
        pass

    def trainModel(self):
        self.mTestMatrix = numpy.random.normal(0.0, 1.0, (self.mDimOutput, self.mDimInput))

    def getResults(self):
        output = numpy.dot(self.mTestMatrix, self.mTrainInput)
        trainAccuracy = Utils.compare(Utils.getPredictions(output), self.mTrainOutput)

        output = numpy.dot(self.mTestMatrix, self.mTestInput)
        testAccuracy = Utils.compare(Utils.getPredictions(output), self.mTestOutput)

        return trainAccuracy, testAccuracy
