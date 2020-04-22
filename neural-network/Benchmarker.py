#!/usr/bin/env python3
import argparse
import json
import logging
import random
import sys

from ModelTrainerExample import ModelTrainerExample
from NeuralNetwork import NeuralNetwork

from MachineLearningUtils import (
    Constants,
    JsonError,
    Utils,
)

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
LOGGER.addHandler(logging.StreamHandler(sys.stdout))

class Benchmarker():
    JSON_DATASET_KEY = "dataset"
    JSON_TRAINER_KEY = "trainer"
    JSON_TRAINER_CONFIG_KEY = "trainer_config"
    JSON_TRAIN_ROUNDS_KEY = "train_rounds"
    JSON_TRAIN_EXAMPLE_PERCENTAGE_KEY = "train_percentage"

    def __init__(self):
        self.mTrainRounds = 1
        self.mTrainPercentage = 70.0
        self.mDataset = Constants.DATASET_MNIST
        self.mDataImages = "train-images-idx3-ubyte"
        self.mDataLabels = "train-labels-idx1-ubyte"
        self.mModelName = ""
        self.mModleConfig = ""

    def parseConfigs(self, configFile: str):
        LOGGER.info("configuration file: " + configFile)
        try:
            f = open(configFile, 'r')
        except IOError:
            LOGGER.exception(f"Failed to load {configFile}")

        with f:
            try:
                configs = json.loads(f.read())

                if self.JSON_DATASET_KEY in configs:
                    self.mDataset = configs[self.JSON_DATASET_KEY]
                    LOGGER.info(f"dataset: {self.mDataset}")
                else:
                    LOGGER.info(f"using default train_rounds: {self.mDataset}")
                if self.mDataset == Constants.DATASET_MNIST:
                    self.mDataImages = "train-images-idx3-ubyte"
                    self.mDataLabels = "train-labels-idx1-ubyte"
                else:
                    LOGGER.error(f"{self.mDataset} is not existed")

                if self.JSON_TRAINER_KEY in configs:
                    self.mModelName = configs[self.JSON_TRAINER_KEY]
                    LOGGER.info(f"trainer: {self.mModelName}")
                else:
                    LOGGER.error(f"failed to parse key: {self.JSON_TRAINER_KEY}")
                    raise JsonError

                if self.JSON_TRAINER_CONFIG_KEY in configs:
                    self.mModelConfig = configs[self.JSON_TRAINER_CONFIG_KEY]
                    LOGGER.info(f"trainer_config: {self.mModelConfig}")
                else:
                    LOGGER.error(f"failed to parser key: {self.JSON_TRAINER_CONFIG_KEY}")
                    raise JsonError

                if self.JSON_TRAIN_ROUNDS_KEY in configs:
                    self.mTrainRounds = configs[self.JSON_TRAIN_ROUNDS_KEY]
                    LOGGER.info(f"train_rounds: {self.mTrainRounds}")
                else:
                    LOGGER.info(f"using default train_rounds: {self.mTrainRounds}")

                if self.JSON_TRAIN_EXAMPLE_PERCENTAGE_KEY in configs:
                    self.mTrainPercentage = configs[self.JSON_TRAIN_EXAMPLE_PERCENTAGE_KEY]
                    LOGGER.info(f"train_percentage: {self.mTrainPercentage}")
                else:
                    LOGGER.info(f"using default train_rounds: {self.mTrainPercentage}")

            except JsonError:
                LOGGER.exception("Invalid json file")

    def generateMNISTSamples(self):
        try:
            inputImages = open(self.mDataImages, "rb")
            inputLables = open(self.mDataLabels, "rb")
            trainImagesFile = open(Constants.FILE_TRAIN_INPUT, "w+b")
            trainLablesFile = open(Constants.FILE_TRAIN_LABELS, "w+b")
            testImagesFile = open(Constants.FILE_TEST_INPUT, "w+b")
            testLablesFile = open(Constants.FILE_TEST_LABELS, "w+b")
        except IOError:
            LOGGER.Exception("Failed to open files in generateMNISTSamples")

        try:
            imagesData = inputImages.read()
            labelsData = inputLables.read()
            numImages = Utils.bytesToInt(imagesData, 4)
            numLabels = Utils.bytesToInt(labelsData, 4)
            if not numImages == numLabels:
                raise ValueError(f"numImages: {numImages} != numLabels: {numLabels}")

            LOGGER.info(f"number of samples in {self.mDataImages}: {numImages}")
            rows = Utils.bytesToInt(imagesData, 8)
            cols = Utils.bytesToInt(imagesData, 12)
            pixels = rows * cols
            if rows != 28 or cols != 28:
                raise ValueError(f"rows: {rows} is not 28, cols: {cols} is not 28")

            numTrainSamples = int(numImages * self.mTrainPercentage / 100)
            numTestSamples = numImages - numTrainSamples
            LOGGER.info(f"dimention for each sample: {pixels}, total samples: {numTrainSamples}")

            samples = random.sample(range(0, numImages), numTrainSamples)
            samples.sort()

            trainIndex = 0
            imageOffset = 16
            labelOffset = 8

            testImagesFile.write(numTestSamples.to_bytes(4, byteorder="big"))
            testImagesFile.write(pixels.to_bytes(4, byteorder="big"))
            testLablesFile.write(numTestSamples.to_bytes(4, byteorder="big"))

            trainImagesFile.write(numTrainSamples.to_bytes(4, byteorder="big"))
            trainImagesFile.write(pixels.to_bytes(4, byteorder="big"))
            trainLablesFile.write(numTrainSamples.to_bytes(4, byteorder="big"))

            for index in range(0, numImages):
                imageFileOffset = index * pixels + imageOffset
                if (trainIndex == numTrainSamples or \
                        index < samples[trainIndex]):
                    testImagesFile.write(bytearray(imagesData[imageFileOffset:imageFileOffset+pixels]))
                    testLablesFile.write(labelsData[labelOffset + index].to_bytes(1, byteorder="big"))
                elif (index == samples[trainIndex]):
                    trainImagesFile.write(bytearray(imagesData[imageFileOffset:imageFileOffset+pixels]))
                    trainLablesFile.write(labelsData[labelOffset + index].to_bytes(1, byteorder="big"))
                    trainIndex = trainIndex + 1

        except ValueError:
            LOGGER.Exception("Failed to generate train samples")
        except IOError:
            LOGGER.Exception("Failed to write data to files")

    def generateSamples(self):
        if self.mDataset == Constants.DATASET_MNIST:
            self.generateMNISTSamples()

    def generateTrainer(self):
        if self.mModelName == Constants.TEST_TRAINER:
            return ModelTrainerExample(self.mDataset, self.mModelConfig)
        elif self.mModelName == Constants.NEURAL_NETWORK_TRAINER:
            return NeuralNetwork(self.mDataset, self.mModelConfig)
        else:
            raise ValueError(f"No existing trainer {self.mModelName}")

    def trainModel(self):
        LOGGER.info("Traininig model")
        self.mTrainAccuracy = []
        self.mTestAccuracy = []
        for trainRound in range(0, self.mTrainRounds):
            print(f"round {trainRound}:")
            self.generateSamples()
            trainer = self.generateTrainer()
            trainer.trainModel()
            trainAccuracy, testAccuracy = trainer.getResults()
            print(f"training accuracy: {trainAccuracy}, test accuracy: {testAccuracy}")
            self.mTrainAccuracy.append(trainAccuracy)
            self.mTestAccuracy.append(testAccuracy)

    def report(self):
        LOGGER.info("Reporting")
        print(f"{self.mModelName}:")
        print(f"trainingset avg accuracy: {sum(self.mTrainAccuracy)/len(self.mTrainAccuracy)}")
        print(f"test avg accuracy: {sum(self.mTestAccuracy)/len(self.mTestAccuracy)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="configuration file in json format", type=str)
    args = parser.parse_args()
    trainer = Benchmarker()
    if args.config:
        trainer.parseConfigs(args.config)
    else:
        LOGGER.error("No valid configuration file")
        exit()
    trainer.trainModel()
    trainer.report()
