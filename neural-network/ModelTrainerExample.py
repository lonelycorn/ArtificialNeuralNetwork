#!/usr/bin/env python3
import argparse
import json
import logging
import numpy
import sys

from ModelTrainerBase import ModelTrainerBase

from MachineLearningUtils import (
    Constants,
    Utils,
)

class ModelTrainerExample(ModelTrainerBase):
    def trainModel(self):
        '''
        override
        '''
        self.mTestMatrix = numpy.random.normal(0.0, 1.0, (self.mDimOutput, self.mDimInput))

    def getResults(self):
        '''
        override
        '''
        output = numpy.dot(self.mTestMatrix, self.mTrainInput)
        trainAccuracy = Utils.compare(Utils.getPredictions(output), self.mTrainOutput)

        output = numpy.dot(self.mTestMatrix, self.mTestInput)
        testAccuracy = Utils.compare(Utils.getPredictions(output), self.mTestOutput)

        return trainAccuracy, testAccuracy
