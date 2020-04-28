#!/usr/bin/env python3
import argparse
import json
import logging
import numpy
import random
import sys

import ActivationFunction
import CostFunction

from MachineLearningUtils import (
    Constants,
    Utils,
)
from ModelTrainerExample import ModelTrainerExample


class NeuralLayer:
    def __init__(self, numCurrentNeurons, numNextNeurons = 0):
        self.mNextLayer = None
        self.mPrevLayer = None

        # input of current layer
        self.mActivations = None
        # input*weights + bias, the sums before activation of next layer
        self.mSums = None

        if numNextNeurons == 0:
            self.isOutput = True
        else:
            self.mBias = numpy.zeros((1, numNextNeurons))
            std = numpy.sqrt(2.0 / numCurrentNeurons)
            self.mWeights = numpy.random.normal(0.0, std, (numCurrentNeurons, numNextNeurons))
            print(f"{numNextNeurons}, {numCurrentNeurons}")

    def feedForward(self, activationFunc):
        assert(self.mActivations is not None)
        assert(self.mNextLayer is not None)
        self.mSums = numpy.dot(self.mActivations, self.mWeights) + self.mBias
        self.mNextLayer.mActivations = activationFunc.getValue(self.mSums)
        # print(f"feedforward: bias {self.mBias.shape}, weights {self.mWeights.shape}")
        # print(f"feedforward: activations {self.mActivations.shape}, sums {self.mSums.shape}")

    def backPropagate(self, activationFunc):
        assert(self.mNextLayer is not None)
        self.mBiasGradient = activationFunc.getDerivative(self.mSums) * \
            numpy.dot(self.mNextLayer.mBiasGradient, self.mNextLayer.mWeights.transpose())
        self.mWeightsGradient = numpy.dot(self.mActivations.transpose(), self.mBiasGradient)

    def backPropagateOutput(self, truths, costFunc, activationFunc):
        self.mBiasGradient = costFunc.getDerivative(truths, self.mNextLayer.mActivations) * \
            activationFunc.getDerivative(self.mSums)
        self.mWeightsGradient = \
            numpy.dot(self.mActivations.transpose(), self.mBiasGradient)

    def update(self, learningRate):
        # print(f"update(): mBiasGradient shape {self.mBiasGradient.shape}, sum {numpy.sum(self.mBiasGradient, axis=0).shape}")
        # print(f"update(): mWeightsGradient shape {self.mWeightsGradient.shape}, sum {numpy.sum(self.mWeightsGradient, axis=0).shape}")
        self.mBias = self.mBias - numpy.sum(self.mBiasGradient, axis=0) * learningRate
        self.mWeights = self.mWeights - self.mWeightsGradient * learningRate
        # self.mWeights = self.mWeights - numpy.sum(self.mWeightsGradient, axis=0) * learningRate

class NeuralNetwork(ModelTrainerExample):
    JSON_EPOCHES_KEY = "epoches"
    JSON_HIDDEN_LAYERS_KEY = "hidden_layers"
    JSON_LEARNING_RATE_KEY = "learning_rate"
    JSON_BATCH_SIZE_KEY = "batch_size"
    JSON_ACTIVATION_FUNCTION_KEY = "activation_function"

    def initLayers(self, sizes):
        sizes.append(0)
        self.mLayers = []

        for (size, nextSize) in zip(sizes[:-1], sizes[1:]):
            self.mLayers.append(NeuralLayer(size, nextSize))

        for (layer, nextLayer) in zip(self.mLayers[:-1], self.mLayers[1:]):
            layer.mNextLayer = nextLayer

        for (layer, prevLayer) in zip(self.mLayers[1:], self.mLayers[:-1]):
            layer.mPrevLayer = prevLayer

    def feedForward(self, input):
        self.mLayers[0].mActivations = input
        for layer in self.mLayers[:-1]:
            layer.feedForward(self.mActivationFunc)

        return self.mLayers[-1].mActivations

    def backPropagate(self, truths):
        self.mLayers[-2].backPropagateOutput(truths, self.mCostFunc, self.mActivationFunc)

        for index in range(len(self.mLayers) - 3, -1, -1):
            self.mLayers[index].backPropagate(self.mActivationFunc)

    def updateNetwork(self, learningRate):
        for layer in self.mLayers[:-1]:
            layer.update(learningRate)

    def train(self):
        inputBatches, outputBatches = Utils.generateBatches(
            self.mBatchSize, self.mTrainInput, self.mTrainOutput
        )
        print(self.mBatchSize)
        print(len(inputBatches))
        '''
        biasGradient = []
        weightsGradient = []
        for layer in self.mLayers:
            biasGradient.append(numpy.zeros(layer.mBias.shape()))
            weightsGradient.append(numpy.zero(layer.mWeights.shape()))
        '''

        for (input, output) in zip(inputBatches, outputBatches):
            self.feedForward(input)
            self.backPropagate(output)
            rate = self.mLearningRate / self.mBatchSize
            self.updateNetwork(rate)
            '''
            for (biasDelta, weightsDelta, layer) in zip(biasGradient, weightsGradient, self.mLayers):
                biasDelta += layer.mBiasGradient
                weightsDelta += layer.mWeightsGradient
            '''

    def parseConfig(self, config):
        '''
        override
        '''
        try:
            f = open(config, 'r')
        except IOError:
            LOGGER.exception(f"Failed to load {configFile}")

        with f:
            configs = json.loads(f.read())
            if self.JSON_EPOCHES_KEY in configs:
                self.mEpoches = configs[self.JSON_EPOCHES_KEY]
            else:
                self.mEpoches = 100

            if self.JSON_LEARNING_RATE_KEY in configs:
                self.mLearningRate = configs[self.JSON_LEARNING_RATE_KEY]
            else:
                self.mLearningRate = 0.1

            if self.JSON_BATCH_SIZE_KEY in configs:
                self.mBatchSize = configs[self.JSON_BATCH_SIZE_KEY]
            else:
                self.mBatchSize = 10

            if self.JSON_ACTIVATION_FUNCTION_KEY in configs:
                activation = configs[self.JSON_ACTIVATION_FUNCTION_KEY]
                if activation == Constants.SIGMOID:
                    self.mActivationFunc = ActivationFunction.Sigmoid()
                elif activation == Constants.RELU:
                    self.mActivationFunc = ActivationFunction.Relu()
                else:
                    raise ValueError(f"No activation function {activation}")
            else:
                self.mActivationFunc = ActivationFunction.Sigmoid()

            self.mCostFunc = CostFunction.Quadratic()

            sizes = []
            sizes = [self.mDimInput]
            if self.JSON_HIDDEN_LAYERS_KEY in configs:
                for hiddenLayer in configs[self.JSON_HIDDEN_LAYERS_KEY]:
                    sizes.append(hiddenLayer)
            sizes.append(self.mDimOutput)
            self.initLayers(sizes)

    def trainModel(self):
        '''
        override
        '''
        for epoch in range(0, self.mEpoches):
            print(f"epoch: {epoch}")
            self.train()
            output = self.feedForward(self.mTrainInput)
            trainAccuracy = Utils.compare(Utils.getPredictions(output), self.mTrainOutput)
            print(f"training accuracy: {trainAccuracy}")


    def getResults(self):
        '''
        override
        '''
        output = self.feedForward(self.mTrainInput)
        trainAccuracy = Utils.compare(Utils.getPredictions(output), self.mTrainOutput)

        output = self.feedForward(self.mTestInput)
        testAccuracy = Utils.compare(Utils.getPredictions(output), self.mTestOutput)
        return trainAccuracy, testAccuracy
