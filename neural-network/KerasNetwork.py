from keras.models import Sequential
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten

import numpy as np
import json
from MachineLearningUtils import Utils

class Network:
    def __init__(self):
        self.mModel = None
        self.mFitArgs = None


    def initialize(self, jsonStr):
        """
        JSON format:
        {
            "layers": array of Layer,
            "compile_args": {
                "loss": str
                "optimizer": str
                "metrics": array of str
            },
            "fit_args": {
                "batch_size": int
                "epoch": int
            }
        }

        where:
        Layer:
        {
            "type": str ("Dense", "Conv2D", "MaxPooling2D")
            **kwargs: per layer definition
        }

        """
        root = json.loads(jsonStr)
        if (not isinstance(root, dict)):
            raise InputError("Invalid JSON format")
        layers = root["layers"]
        compileArgs = root["compile_args"]
        fitArgs = root["fit_args"]

        # build the network
        self.mModel = Sequential()
        self.mLayerTags = []
        supportedLayers = { l.__name__: l for l in \
                [Input, Dense, Conv2D, MaxPooling2D, Flatten]
        }
        for l in layers:
            layerType = l["type"]
            if (layerType not in supportedLayers):
                raise ValueError(f"Unsupported layer type {layerType}")
            l.pop("type")
            self.mModel.add(supportedLayers[layerType](**l))
            self.mLayerTags.append(layerType)

        # compile the network
        self.mModel.compile(**compileArgs)

        # remember the fit args
        self.mFitArgs = fitArgs


    def train(self, xs, ys):
        self.mModel.fit(xs, ys, **self.mFitArgs)


    def validate(self, xs, ys):
        yHat = self.mModel.predict(xs)
        labels = Utils.getPredictions(yHats)
        accuracy = Utils.compare(labels, ys)
        return accuracy

    def print(self):
        print("[KerasNetwork]")
        print(self.mModel.summary())


if (__name__ == "__main__"):
    n = Network()

    # MNIST data and labels
    with open("train-images-idx3-ubyte", "rb") as f1, open("train-labels-idx1-ubyte", "rb") as f2:
        # image file format
        #   - int32_t magic number
        #   - int32_t image count
        #   - int32_t image rows
        #   - int32_t image cols
        #   - uint8_t * N row-wise pixels
        # label file format
        #   - int32_t magic number
        #   - int32_t label count
        #   - uint8_t * N lables
        imageRaw = f1.read()
        labelRaw = f2.read()

        imageOffset = 16
        labelOffset = 8

        images = np.frombuffer(imageRaw, dtype=np.uint8, offset=imageOffset)
        labels = np.frombuffer(labelRaw, dtype=np.uint8, offset=labelOffset)

        xs = images.reshape(-1, 1, 28, 28) / 255.0
        ys = np.zeros((len(xs), 10))
        for (i, l) in enumerate(labels):
            ys[i, l] = 1.0


    with open("keras-network-mnist.json", "r") as f:
        jsonStr = f.read()
        n.initialize(jsonStr)

    n.print()
    n.train(xs, ys)

