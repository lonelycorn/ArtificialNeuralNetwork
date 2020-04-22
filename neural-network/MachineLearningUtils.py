#!/usr/bin/env python3
import numpy

class JsonError(Exception):
    """Raised when fail to parse configuration file"""
    pass

class Constants:
    DATASET_MNIST = "MNIST"
    FILE_TRAIN_INPUT = "train_input"
    FILE_TRAIN_LABELS = "train_labels"
    FILE_TEST_INPUT = "test_input"
    FILE_TEST_LABELS = "test_labels"
    TEST_TRAINER = "trainer_example"
    NEURAL_NETWORK_TRAINER = "neural-network"
    SIGMOID = "sigmoid"
    RELU = "relu"

class Utils:
    @staticmethod
    def bytesToInt(data, start):
        '''
        Convert 4 bytes to an int
        @param data: the data of input
        @param start: the start index in data
        @return: an int
        '''
        return (data[start] << 24) + (data[start + 1] << 16) + \
            (data[start + 2] << 8) + data[start + 3]

    @staticmethod
    def compare(predictions, truths):
        '''
        Compare the matrix of predictions and truths
        @param predictions: the matirx of prediction labels; each row is a label
        @param truths: the matrix of truth labels; each row is a label
        @return: accuracy of predictions
        '''
        assert(predictions.shape == truths.shape)
        bingo = 0
        for (p, t) in zip(predictions, truths):
            if (numpy.array_equal(p, t)):
                bingo += 1
        n = len(predictions)
        return 1.0 * bingo / n

    @staticmethod
    def getPredictions(possibilities):
        '''
        Get prediction labels accroding to the possibility of each output
        Use the max value in the possibility
        @param possibility: the matrix of possibility of each sample output
        @return: the prediction labels
        '''
        predictions = numpy.zeros(possibilities.shape)
        for (i, p) in enumerate(possibilities):
            predictions[i, numpy.argmax(p)] = 1
        return predictions

    @staticmethod
    def generateBatches(size, input, output):
        '''
        generate random batches based on the size
        @param size: size
        @param input: input samples
        @param output: output samples
        @return inputBatches, outputBatches
        '''
        inputBatches = []
        outputBatches = []
        randomIndex = numpy.arange(len(input))
        numpy.random.shuffle(randomIndex)
        input = input[randomIndex]
        output = output[randomIndex]
        for index in range(size, len(input) + 1, size):
            inputBatches.append(input[index-size: index])
            outputBatches.append(output[index-size: index])
        return inputBatches, outputBatches
