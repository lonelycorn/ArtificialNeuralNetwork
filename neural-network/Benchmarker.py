#!/usr/bin/env python3
import argparse
import json
import logging
import random
import sys

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
LOGGER.addHandler(logging.StreamHandler(sys.stdout))

class JsonError(Exception):
    """Raised when fail to parse configuration file"""
    pass

class Benchmarker():
    FILE_TRAINING_IMAGES = "training_images"
    FILE_TRAINING_LABELS = "training_labels"
    FILE_TEST_IMAGES = "test_images"
    FILE_TEST_LABELS = "test_labels"

    JSON_DATASET_KEY = "dataset"
    JSON_TRAINER_KEY = "trainer"
    JSON_TRAINER_CONFIG_KEY = "trainer_config"
    JSON_TRAIN_ROUNDS_KEY = "train_rounds"
    JSON_TRAINING_EXAMPLE_PERCENTAGE_KEY = "training_percentage"

    DATASET_MNIST = "MNIST"

    def __init__(self):
        self.mTrainRounds = 1
        self.mTrainingPercentage = 70.0
        self.mDataset = self.DATASET_MNIST
        self.mDataImages = "train-images-idx3-ubyte"
        self.mDataLabels = "train-labels-idx1-ubyte"
        self.mModel = ""
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
                if self.mDataset == self.DATASET_MNIST:
                    self.mDataImages = "train-images-idx3-ubyte"
                    self.mDataLabels = "train-labels-idx1-ubyte"
                else:
                    LOGGER.error(f"{self.mDataset} is not existed")


                if self.JSON_TRAINER_KEY in configs:
                    self.mModel = configs[self.JSON_TRAINER_KEY]
                    LOGGER.info(f"trainer: {self.mModel}")
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

                if self.JSON_TRAINING_EXAMPLE_PERCENTAGE_KEY in configs:
                    self.mTrainingPercentage = configs[self.JSON_TRAINING_EXAMPLE_PERCENTAGE_KEY]
                    LOGGER.info(f"training_percentage: {self.mTrainingPercentage}")
                else:
                    LOGGER.info(f"using default train_rounds: {self.mTrainingPercentage}")

            except JsonError:
                LOGGER.exception("Invalid json file")

    def bytesToInt(self, data, start):
        return (data[start] << 24) + (data[start + 1] << 16) + \
                (data[start + 2] << 8) + data[start + 3]

    def generateMNISTSamples(self):
        try:
            inputImages = open(self.mDataImages, "rb")
            inputLables = open(self.mDataLabels, "rb")
            trainImagesFile = open(self.FILE_TRAINING_IMAGES, "w+b")
            trainLablesFile = open(self.FILE_TRAINING_LABELS, "w+b")
            testImagesFile = open(self.FILE_TEST_IMAGES, "w+b")
            testLablesFile = open(self.FILE_TEST_LABELS, "w+b")
        except IOError:
            LOGGER.Exception("Failed to open files in generateMNISTSamples")

        try:
            imagesData = inputImages.read()
            labelsData = inputLables.read()
            numImages = self.bytesToInt(imagesData, 4)
            numLabels = self.bytesToInt(labelsData, 4)
            if not numImages == numLabels:
                raise ValueError(f"numImages: {numImages} != numLabels: {numLabels}")

            LOGGER.info(f"number of samples in {self.mDataImages}: {numImages}")
            rows = self.bytesToInt(imagesData, 8)
            cols = self.bytesToInt(imagesData, 12)
            pixels = rows * cols
            if rows != 28 or cols != 28:
                raise ValueError(f"rows: {rows} is not 28, cols: {cols} is not 28")

            numTrainingSamples = int(numImages * self.mTrainingPercentage / 100)
            numTestSamples = numImages - numTrainingSamples
            LOGGER.info(f"dimention for each sample: {pixels}, total samples: {numTrainingSamples}")

            samples = random.sample(range(0, numImages), numTrainingSamples)
            samples.sort()

            trainingIndex = 0
            imageOffset = 16
            labelOffset = 8

            testImagesFile.write(numTestSamples.to_bytes(4, byteorder="big"))
            testImagesFile.write(pixels.to_bytes(4, byteorder="big"))
            testLablesFile.write(numTestSamples.to_bytes(4, byteorder="big"))

            trainImagesFile.write(numTrainingSamples.to_bytes(4, byteorder="big"))
            trainImagesFile.write(pixels.to_bytes(4, byteorder="big"))
            trainLablesFile.write(numTrainingSamples.to_bytes(4, byteorder="big"))

            for index in range(0, numImages):
                imageFileOffset = index * pixels + imageOffset
                if (trainingIndex == numTrainingSamples or \
                        index < samples[trainingIndex]):
                    testImagesFile.write(bytearray(imagesData[imageFileOffset:imageFileOffset+pixels]))
                    testLablesFile.write(labelsData[labelOffset + index].to_bytes(1, byteorder="big"))
                elif (index == samples[trainingIndex]):
                    trainImagesFile.write(bytearray(imagesData[imageFileOffset:imageFileOffset+pixels]))
                    trainLablesFile.write(labelsData[labelOffset + index].to_bytes(1, byteorder="big"))
                    trainingIndex = trainingIndex + 1

        except ValueError:
            LOGGER.Exception("Failed to generate training samples")
        except IOError:
            LOGGER.Exception("Failed to write data to files")

    def generateSamples(self):
        if self.mDataset == self.DATASET_MNIST:
            self.generateMNISTSamples()

    def trainModel(self):
        LOGGER.info("Training")
        for trainRound in range(0, self.mTrainRounds):
            LOGGER.info(f"round {trainRound}")
            self.generateSamples()

    def report(self):
        LOGGER.info("Reporting")


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
