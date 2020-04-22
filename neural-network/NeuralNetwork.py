#!/usr/bin/env python3
import argparse
import json
import logging
import numpy
import random
import sys

def SigmoidActivationFunc(vector):
    return 1 / (1 + numpy.exp(-vector))

class NeuralLayer:
    def __init__(self, numCurrentNeurons, numNextNeurons):
        self.mNextLayer = None
        self.mPrevLayer = None

        self.mActivations = None
        self.mSums = None

        if numNextNeurons == 0:
            self.isOutput = True
        else:
            self.mBias = numpy.random.normal(0.0, 1.0, (numNextNeurons, 1))
            self.mWeights = numpy.random.normal(0.0, 1.0, (numNextNeurons, numCurrentNeurons))
            print(f"{numNextNeurons}, {numCurrentNeurons}")

    def feedForward(self, activationFunc):
        assert(self.mActivations is not None)
        assert(self.mNextLayer is not None)
        self.mSums = numpy.dot(self.mWeights, self.mActivations) + self.mBias
        self.mNextLayer.mActivations = activationFunc(self.mSums)

    def backPropagate(self, activationDerivative):
        assert(self.mPrevLayer is not None)
        assert(self.mNextLayer is not None)
        self.mBiasGradient = activationDerivative(self.mSums) * \
            numpy.dot(self.mNextLayer.mWeights.transpose(), self.mNextLayer.mBiasGradient)
        self.mWeightsGradient = numpys.dot(self.mBiasGradient, self.mPrevLayer.mActivations.transpose())

class NeuralNetwork:
    def __init__(self, inputDimension, outputDimension, hiddenLayerDimension):
        self.activationFunc = SigmoidActivationFunc
        self.layers = []
        self.numLayers = len(hiddenLayerDimension) + 2

        if self.numHiddenLayers == 0:
            self.inputLayer = NeuralLayer(inputDimension, self.outputLayer)
        else:
            self.hiddenLayers.insert(0, NeuralLayer(hiddenLayerDimension[-1], self.outputLayer))
            for index in range(self.numHiddenLayers - 2, -1, -1):
                self.hiddenLayers.insert(0, NeuralLayer(hiddenLayerDimension[index], self.hiddenLayers[0]))
            self.inputLayer = NeuralLayer(inputDimension, self.hiddenLayers[0])

    def SetActivationFunction(self, func):
        self.activationFunc = func

    def GetOutput(self, inputSample):
        self.inputLayer.activations = inputSample
        self.inputLayer.FeedForward(self.activationFunc)
        for layer in self.hiddenLayers:
            layer.FeedForward(self.activationFunc)
        self.outputLayer.activations = numpy.sum(self.outputLayer.activations)
        return self.outputLayer.activations

    def TrainBackPropagate(self, standardOutput):
        if self.numHiddenLayers == 0:
            self.outputLayer.BackPropagate(self.inputLayer)
        else:
            self.outputLayer.BackPropagate(self.hiddenLayers[-1])
            for index in range(self.numHiddenLayers - 1, 0, -1):
                self.hiddenLayers[index].BackPropagate()

def BytesToInt(data, start):
    return (data[start] << 24) + (data[start + 1] << 16) + \
            (data[start + 2] << 8) + data[start + 3]

if __name__ == '__main__':
    data = numpy.fromfile("train_input", dtype=numpy.uint8)
    num = BytesToInt(data, 0)
    dimension = bytesToInt(data, 4)
    print(f"{num}, {dimension}")
    network = NeuralNetwork(dimension, 10, [20, 25])
    data = data[8:]
    data.shape = (num, dimension)

    sample = data[0:1,:]
    print(sample.shape)
    sample = sample.T / 255.0
    print(sample.shape)
    print(network.GetOutput(sample))
