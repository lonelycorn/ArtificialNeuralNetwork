#!/usr/bin/env python3
import argparse
import json
import logging
import numpy
import sys

from MachineLearningUtils import (
    Constants,
    Utils,
)

class ModelTrainerBase():
    def __init__(self, dataset: str, config: str):
        self.initDataset(dataset)
        self.parseConfig(config)

    def initDataset(self, dataset: str):
        '''
        interface
        '''
        raise NotImplementedError("initDataset() was not implemented")

    def parseConfig(self, config: str):
        '''
        interface
        '''
        pass

    def trainModel(self):
        '''
        interface
        '''
        raise NotImplementedError("trainModel() was not implemented")

    def getResults(self):
        '''
        interface
        '''
        raise NotImplementedError("getResults() was not implemented")
