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
    def __init__(self, numCurrentNeurons, nextLayer):
        self.numNeurons = numCurrentNeurons
        self.nextLayer = nextLayer

        self.inputValues = None

        self.isOutput = False
        if nextLayer == None:
            self.isOutput = True
        else:
            self.bias = numpy.random.normal(0.0, 1.0, (self.nextLayer.numNeurons, 1))
            self.weightsMatrix = numpy.random.normal(0.0, 1.0, (nextLayer.numNeurons, self.numNeurons))
            print(f"{nextLayer.numNeurons}, {self.numNeurons}")

    def FeedForward(self, activationFunc, input = None):
        assert(self.activationValues is not None)
        assert(self.nextLayer is not None)
        self.nextLayer.activationValues = activationFunc(
            numpy.dot(self.weightsMatrix, self.activationValues) + self.bias
        )

    def GetActivationValues(self):
        assert(self.activationValues is not None)
        return self.activationValues

class NeuralNetwork:
    def __init__(self, inputDimension, outputDimension, hiddenLayerDimension):
        self.activationFunc = SigmoidActivationFunc
        self.outputLayer = NeuralLayer(outputDimension, None)
        self.hiddenLayers = []
        self.numHiddenLayers = len(hiddenLayerDimension)

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
        self.inputLayer.activationValues = inputSample
        self.inputLayer.FeedForward(self.activationFunc)
        for layer in self.hiddenLayers:
            layer.FeedForward(self.activationFunc)
        self.outputLayer.activationValues = numpy.sum(self.outputLayer.activationValues)
        return self.outputLayer.activationValues

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
    data = numpy.fromfile("training_images", dtype=numpy.uint8)
    num = BytesToInt(data, 0)
    dimension = BytesToInt(data, 4)
    print(f"{num}, {dimension}")
    network = NeuralNetwork(dimension, 10, [20, 25])
    data = data[8:]
    data.shape = (num, dimension)

    sample = data[0:1,:]
    print(sample.shape)
    sample = sample.T / 255.0
    print(sample.shape)
    print(network.GetOutput(sample))
