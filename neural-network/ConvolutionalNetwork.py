import json
import numpy as np
import pickle
import time

import ActivationFunction
import CostFunction
from MachineLearningUtils import Utils

class LayerBase:
    def __init__(self, id, tag):
        self.mId = id
        self.mTag = tag
        self.mInputLayer = None
        self.mOutputLayer = None


    @property
    def id(self):
        return self.mId


    @property
    def tag(self):
        return self.mTag


    def registerInput(self, layer):
        if (self.mInputLayer is not None):
            raise RuntimeError("Input already registered")
        self.mInputLayer = layer


    def registerOutput(self, layer):
        if (self.mOutputLayer is not None):
            raise RuntimeError("Output already registered")


    def getOutputShape(self):
        raise NotImplementedError


    def initialize(self):
        raise NotImplementedError


    def feedforward(self, x):
        """
        :param [in] x: input to the layer
        :return output from the layer
        """
        raise NotImplementedError


    def backPropagate(self, dC_over_da):
        """
        :param [in] dC_over_da: partial derivative of cost w.r.t. layer output
        :return dC_over_dx; partial derivative of cost w.r.t. layer input
        """
        raise NotImplementedError


    def update(self, learningRate):
        raise NotImplementedError


    def print(self):
        print(f"[layer] id = {self.mId}, tag = {self.mTag}")


class SimpleConvolutional2dLayer(LayerBase):
    """
    A simple convolutional layer whose stride == (1, 1)
    """
    @staticmethod
    def getName():
        return "simple_convolutional_2d"


    def __init__(self, id, activationFunction, kernelCount, kernelSize):
        """
        :param [in] kernelCount: an integer; number of unique kernels
        :param [in] kernelSize: a 2-tuple; 2D size of the receptive field
        """
        LayerBase.__init__(self, id, self.getName())

        if (isinstance(activationFunction, str)):
            activationFunction = ActivationFunction.create(activationFunction)
        self.mActivationFunction = activationFunction

        self.mKernelCount = kernelCount
        self.mKernelSize = kernelSize
        self.mKernels = None
        self.mBiases = None

        self.mDeltaKernels = None
        self.mDeltaBiases = None
        self.mSampleCount = 0


    def getOutputShape(self):
        return self.mOutputShape


    def initialize(self):
        if (self.mInputLayer is None):
            raise RuntimeError(f"Layer {self.mId} has no registered input")

        self.mInputShape = self.mInputLayer.getOutputShape()
        (inputDepth, inputRows, inputCols) = self.mInputShape
        (kernelRows, kernelCols) = self.mKernelSize

        if ((inputRows < kernelRows) or (inputCols < kernelCols)):
            raise ValueError("Improper input / kernel size")

        kernelShape = (self.mKernelCount, inputDepth, kernelRows, kernelCols)

        outputRows = inputRows - kernelRows + 1
        outputCols = inputCols - kernelCols + 1
        self.mOutputShape = np.array([self.mKernelCount, outputRows, outputCols])
        print(f"self.mOutputShape = {self.mOutputShape}")

        # first dimension is kernel idx;
        # He initialization
        std = np.sqrt(2.0 / np.prod(kernelShape[1:]))
        self.mKernels = np.random.normal(0.0, std, kernelShape)
        self.mBiases = np.zeros((self.mKernelCount, 1))

        self.mDeltaKernels = np.zeros(self.mKernels.shape)
        self.mDeltaBiases = np.zeros(self.mBiases.shape)
        self.mSampleCount = 0


    def feedforward(self, x):
        #print(f"[{self.mTag}] feedforward\nx.shape={x.shape}")
        if (len(x.shape) == 2):
            x = np.array([x])

        self.mLatestInputs = x

        (kernelRows, kernelCols) = self.mKernelSize
        (outputDepth, outputRows, outputCols) = self.mOutputShape

        z = np.zeros(self.mOutputShape)
        #print(f"bias =\n{self.mBiases.flatten()}")
        #print(f"z.shape = {z.shape}")

        for (k, kernel) in enumerate(self.mKernels):
            for r in range(outputRows):
                for c in range(outputCols):
                    # NOTE kernel (d, w, h) is applied to __ALL__ channels of x
                    z[k, r, c] = np.sum(x[:, r : r + kernelRows, c : c + kernelCols] * kernel) + self.mBiases[k][0]

        (a, self.mLatestActivationDerivatives) = \
                self.mActivationFunction.getValueAndDerivative(z)
        return a


    def backPropagate(self, dC_over_da):

        dC_over_dz = dC_over_da * self.mLatestActivationDerivatives

        (kernelRows, kernelCols) = self.mKernelSize
        (inputDepth, inputRows, inputCols) = self.mInputShape
        (outputDepth, outputRows, outputCols) = self.mOutputShape

        dC_over_dx = np.zeros(self.mInputShape)

        for (k, kernel) in enumerate(self.mKernels):
            for r0 in range(outputRows):
                for c0 in range(outputCols):
                    self.mDeltaKernels[k, :, :, :] += \
                            dC_over_dz[k, r0, c0] * self.mLatestInputs[:, r0 : r0 + kernelRows, c0 : c0 + kernelCols]
                    dC_over_dx[:, r0 : r0 + kernelRows, c0 : c0 + kernelCols] += \
                            dC_over_dz[k, r0, c0] * kernel
            self.mDeltaBiases[k] += np.sum(dC_over_dz[k, :, :])
        self.mSampleCount += 1

        return dC_over_dx


    def update(self, learningRate):
        if (self.mSampleCount > 0):
            k = learningRate / self.mSampleCount
            self.mKernels -= k * self.mDeltaKernels
            self.mBiases -= k * self.mDeltaBiases

        # prepare for the next update
        self.mSampleCount = 0
        self.mDeltaKernels = np.zeros(self.mKernels.shape)
        self.mDeltaBiases = np.zeros(self.mBiases.shape)


    def print(self):
        LayerBase.print(self)
        print(f"    activation function = {self.mActivationFunction.getName()}")
        print(f"    kernel shape = {self.mKernels.shape}")
        print(f"    input shape = {self.mInputShape}")
        print(f"    output shape = {self.mOutputShape}")


class SimpleMaxPooling2dLayer(LayerBase):
    """
    A simple max-pooling layer whose stride == poolSize
    """
    @staticmethod
    def getName():
        return "simple_max_pooling_2d"


    def __init__(self, id, poolSize):
        """
        :param [in] poolSize: a 2-tuple
        :param [in] stride: a 2-tuple
        """
        LayerBase.__init__(self, id, self.getName())
        self.mPoolSize = poolSize

        self.mInputShape = None
        self.mOutputShape = None

        self.mLatestArgmax = None


    def getOutputShape(self):
        return self.mOutputShape


    def initialize(self):
        if (self.mInputLayer is None):
            raise RuntimeError(f"Layer {self.mId} has no registered input")

        self.mInputShape = self.mInputLayer.getOutputShape()
        (inputDepth, inputRows, inputCols) = self.mInputShape
        (poolRows, poolCols) = self.mPoolSize

        if ((inputRows < poolRows) or (inputCols < poolCols) or
            (inputRows % poolRows > 0) or (inputCols % poolCols > 0)):
            raise ValueError("Improper input / pool size")

        outputDepth = inputDepth
        outputRows = int(inputRows / poolRows)
        outputCols = int(inputCols / poolCols)
        self.mOutputShape = np.array([outputDepth, outputRows, outputCols])


    def feedforward(self, x):
        #print(f"[{self.mTag}] feedforward\nx.shape={x.shape}")
        if (len(x.shape) == 2):
            x = np.array([x])

        if (not np.all(self.mInputShape == x.shape)):
            raise ValueError("Improper input shape")

        (outputDepth, outputRows, outputCols) = self.mOutputShape
        (poolRows, poolCols) = self.mPoolSize

        output = np.zeros(self.mOutputShape)
        argmax = np.zeros(self.mInputShape)

        #print(f"{self.getName()} feedforward")
        #print(f"input shape = {x.shape}")
        #print(f"output shape = {output.shape}")

        for (d, src) in enumerate(x):
            #print(f"src.shape = {src.shape}")
            r0 = 0
            for r in range(outputRows):
                c0 = 0
                for c in range(outputCols):
                    k = np.argmax(src[r0 : r0 + poolRows, c0 : c0 + poolCols])
                    posInPool = np.unravel_index(k, self.mPoolSize)
                    posInSrc = np.array(posInPool) + (r0, c0)
                    #print(f"posInSrc = {posInSrc}")
                    #print(src[posInSrc])
                    output[d, r, c] = src[posInSrc[0], posInSrc[1]]
                    argmax[d, posInSrc[0], posInSrc[1]] = 1.0
                    c0 += poolCols
                r0 += poolRows

        self.mLatestArgmax = argmax

        return output


    def backPropagate(self, dC_over_da):

        (poolRows, poolCols) = self.mPoolSize
        dC_over_da = np.repeat(dC_over_da, poolRows, axis=1)
        dC_over_da = np.repeat(dC_over_da, poolCols, axis=2)

        dC_over_dx = dC_over_da * self.mLatestArgmax
        return dC_over_dx


    def update(self, learningRate):
        # nothing to do; pooling layers do not learn
        pass


    def print(self):
        LayerBase.print(self)
        print(f"    pool size = {self.mPoolSize}")
        print(f"    input shape = {self.mInputShape}")
        print(f"    output shape = {self.mOutputShape}")


class MaxPooling2dLayer(LayerBase):
    @staticmethod
    def getName():
        return "max_pooling_2d"


    def __init__(self, id, poolSize, stride):
        """
        :param [in] poolSize: a 2-tuple
        :param [in] stride: a 2-tuple
        """
        LayerBase.__init__(self, id, self.getName())
        self.mPoolSize = poolSize
        self.mStride = stride
        self.mLatestActivation = None
        self.mLatestArgmax = None

        self.mInputShape = None
        self.mOutputShape = None


    def getOutputShape(self):
        return self.mOutputShape


    def initialize(self):
        if (self.mInputLayer is None):
            raise RuntimeError(f"Layer {self.mId} has no registered input")

        self.mInputShape = self.mInputLayer.getOutputShape()
        (inputRows, inputCols) = self.mInputShape
        (poolRows, poolCols) = self.mPoolSize
        (rowStride, colStride) = self.mStride

        if ((0 != ((inputRows - poolRows) % rowStride)) or
            (0 != ((inputCols - poolCols) % colStride))):
            raise ValueError("Improper pool size / stride")

        outputRows = int((inputRows - poolRows) / rowStride) + 1
        outputCols = int((inputCols - poolCols) / colStride) + 1
        self.mOutputShape = np.array([outputRows, outputCols])


    def feedforward(self, x):
        #print(f"[{self.mTag}] feedforward\nx.shape={x.shape}")

        if (len(x.shape) == 2):
            x = np.array([x])

        outputDepth = x.shape[0]
        (outputRows, outputCols) = self.mOutputShape
        (poolRows, poolCols) = self.mPoolSize
        (rowStride, colStride) = self.mStride

        output = np.zeros((outputDepth, outputRows, outputCols))
        argmax = np.zeros((outputDepth, outputRows, outputCols, 2))

        #print("feedforward")
        #print(f"input shape = {x.shape}")
        #print(f"output shape = {output.shape}")

        for (d, src) in enumerate(x):
            r0 = 0
            for r in range(outputRows):
                c0 = 0
                for c in range(outputCols):
                    k = np.argmax(src[r0 : r0 + poolRows, c0 : c0 + poolCols])
                    posInPool = np.unravel_index(k, self.mPoolSize)
                    posInSrc = np.array(posInPool) + (r0, c0)
                    output[d, r, c] = src[posInSrc]
                    argmax[d, r, c, :] = posInsrc
                    c0 += colStride
                r0 += rowStride

        self.mLatestActivation = output
        self.mLatestArgmax = argmax

        return output


    def backPropagate(self, dC_over_da):
        if (len(x.shape) == 2):
            x = np.array([x])

        inputDepth = x.shape[0]
        (inputRows, inputCols) = self.mInputShape
        (outputRows, outputCols) = self.mOutputShape

        retval = np.zeros((inputDepth, inputRows, inputCols))
        #print("backPropagate")
        #print(f"input shape = {dC_over_da.shape}")
        #print(f"output shape = {retval.shape}")

        for (d, src) in enumerate(x):
            for r in range(outputRows):
                for c in range(outputCols):
                    posInSrc = self.mLatestArgmax[d, r, c]
                    retval[d, posInSrc[0], posInSrc[1]] += dC_over_da[d, r, c]

        return retval


    def update(self, learningRate):
        # nothing to do; pooling layers do not learn
        pass


    def print(self):
        LayerBase.print(self)
        print(f"    pool size = {self.mPoolSize}")
        print(f"    stride = {self.mStride}")
        print(f"    input shape = {self.mInputShape}")
        print(f"    output shape = {self.mOutputShape}")


class FullyConnectedLayer(LayerBase):
    @staticmethod
    def getName():
        return "fully_connected"

    def __init__(self, id, activationFunction, neuronCount):
        LayerBase.__init__(self, id, self.getName())

        if (isinstance(activationFunction, str)):
            activationFunction = ActivationFunction.create(activationFunction)
        self.mActivationFunction = activationFunction

        # Need to remember input shape for back propagation
        self.mInputShape = None
        self.mOutputShape = np.array([1, neuronCount])

        self.mWeights = None
        self.mBiases = None

        self.mLatestInputs = None
        self.mLatestActivationDerivatives = None

        self.mDeltaWeights = None
        self.mDeltaBiases = None
        self.mSampleCount = 0


    def getOutputShape(self):
        return self.mOutputShape


    def initialize(self):
        if (self.mInputLayer is None):
            raise RuntimeError(f"Layer {self.mId} has no registered input")

        self.mInputShape = self.mInputLayer.getOutputShape()
        neuronCount = self.mOutputShape[1]
        inputCount = np.prod(self.mInputShape)
        #print(f"self.mInputShape = {self.mInputShape}")
        #print(f"inputCount = {inputCount}")

        # FIXME: He initialization doesn't seem to always work
        std = np.sqrt(2.0 / inputCount)
        self.mWeights = np.random.normal(0.0, std, (inputCount, neuronCount))
        self.mBiases = np.zeros(self.mOutputShape)
        #std = np.sqrt(1.0 / inputCount)
        #self.mWeights = np.random.normal(0.0, std, (inputCount, neuronCount))
        #self.mBiases = np.random.normal(0.0, 1.0, self.mOutputShape)

        self.mDeltaWeights = np.zeros(self.mWeights.shape)
        self.mDeltaBiases = np.zeros(self.mBiases.shape)
        self.mSampleCount = 0


    def feedforward(self, x):
        #print(f"[{self.mTag}] feedforward\nx.shape={x.shape}")
        # (1, inputCount)
        x = x.reshape(1, -1)

        self.mLatestInputs = x

        # (1, neuronCount)
        z = np.matmul(x, self.mWeights) + self.mBiases

        (a, self.mLatestActivationDerivatives) = \
                self.mActivationFunction.getValueAndDerivative(z)

        return a


    def backPropagate(self, dC_over_da):
        # (1, neuronCount)
        # NOTE: element-wise product
        dC_over_dz = dC_over_da * self.mLatestActivationDerivatives

        # (inputCount, neuronCount)
        self.mDeltaWeights += np.matmul(self.mLatestInputs.T, dC_over_dz)
        # (1, neuronCount)
        self.mDeltaBiases += dC_over_dz
        self.mSampleCount += 1

        # (1, inputCount)
        dC_over_dx = np.matmul(dC_over_dz, self.mWeights.T)

        return dC_over_dx.reshape(self.mInputShape)


    def update(self, learningRate):
        if (self.mSampleCount > 0):
            k = learningRate / self.mSampleCount
            self.mWeights -= k * self.mDeltaWeights
            self.mBiases -= k * self.mDeltaBiases

        # prepare for the next update
        self.mSampleCount = 0
        self.mDeltaWeights = np.zeros(self.mWeights.shape)
        self.mDeltaBiases = np.zeros(self.mBiases.shape)


    def print(self):
        LayerBase.print(self)
        print(f"    activation function = {self.mActivationFunction.getName()}")
        print(f"    neuron count = {len(self.mBiases)}")
        print(f"    input shape = {self.mInputShape}")
        print(f"    output shape = {self.mOutputShape}")


class Input2dLayer(LayerBase):
    @staticmethod
    def getName():
        return "input_2d"


    def __init__(self, id, inputShape):
        """
        :param [in] inputShape: (depth, width, height)
        """
        LayerBase.__init__(self, id, self.getName())
        N = len(inputShape)
        if (N == 2):
            self.mOutputShape = np.array([1, inputShape[0], inputShape[1]])
        elif (N == 3):
            self.mOutputShape = np.array(inputShape)
        else:
            raise ValueError(f"Improper input shape to layer {self.mId}")


    def getOutputShape(self):
        return self.mOutputShape


    def initialize(self):
        pass


    def feedforward(self, x):
        #print(f"[{self.mTag}] feedforward\nx.shape={x.shape}")
        if (len(x.shape) == 2):
            x = np.array([x])

        return x


    def backPropagate(self, dC_over_da):
        pass


    def update(self, learningRate):
        pass


    def print(self):
        LayerBase.print(self)
        print(f"    output shape = {self.mOutputShape}")


class Network:
    def __init__(self):
        # all layers including the input layer
        self.mLayers = []

        self.mCostFunction = None

        self.mBatchSize = 1
        self.mLearningRate = 0.1
        self.mMaxEpoch = 30

    def saveToFile(self, filename):
        print(f"saving to file '{filename}'")
        with open(filename, "wb") as f:
            pickle.dump(self.__dict__, f)
        print("done")


    def loadFromFile(self, filename):
        print(f"loading from file '{filename}'")
        with open(filename, "rb") as f:
            d = pickle.load(f)
            self.__dict__.update(d)
        print("done")


    def initialize(self, jsonStr):
        """
        JSON format:
        {
            "network": {
                "layers": array of Layer
                "edges": array of Edge,
                "costFunction": str
            },
            "hyperparameters": {
                "learningRate": float
                "batchSize": int
                "maxEpoch": int
            }
        }

        where:
        Layer:
        {
            "id": int ( must be unique )
            "tag": str ( "input_2d", "simple_convolutional_2d", "simple_max_pooling_2d", "fully_connected" )
            **kwargs: per layer definition
        }

        Edge:
            [ int, int ]
        ID of the from source layer and estination layer, respectively

        cost_function: one of ( "quadratic", "cross_entropy" )

        """
        root = json.loads(jsonStr)
        if (not isinstance(root, dict)):
            raise InputError("Invalid JSON format")
        network = root["network"]
        hyperparameters = root["hyperparameters"]

        # parse network
        layers = {}
        definedLayers = \
        { l.getName() : l for l in
            [
                SimpleConvolutional2dLayer,
                SimpleMaxPooling2dLayer,
                MaxPooling2dLayer,
                FullyConnectedLayer,
                Input2dLayer
            ]
        }

        for l in network["layers"]:
            id = int(l["id"])
            if (id in layers):
                raise InputError(f"Duplicated layer id {id}")

            tag = l["tag"]
            l.pop("tag")
            if (tag not in definedLayers):
                raise ValueError(f"Unsupported layer tag {tag}")
            else:
                layers.update({ id: definedLayers[tag](**l) })

        edges = network["edges"]
        indegrees = [set() for i in range(len(layers))]
        for e in edges:
            src = e[0]
            dst = e[1]
            layers[src].registerOutput(layers[dst])
            layers[dst].registerInput(layers[src])

            indegrees[dst].add(src)


        self.mCostFunction = CostFunction.create(network["costFunction"])

        # parse hyperparameters
        self.mLearningRate = float(hyperparameters["learningRate"])
        self.mBatchSize = int(hyperparameters["batchSize"])
        self.mMaxEpoch = int(hyperparameters["maxEpoch"])


        # topological sort
        layerCount = len(layers)
        processed = [ False for i in range(layerCount) ]
        self.mLayers.clear()
        for i in range(layerCount):
            id = None
            for j in layers.keys():
                if ((not processed[j]) and (len(indegrees[j]) == 0)):
                    id = j
                    break
            if (id is None):
                raise RuntimeError("Not a valid network")

            processed[id] = True
            self.mLayers.append(layers[id])
            for j in layers.keys():
                indegrees[j].discard(id)

        # initialize layers
        for l in self.mLayers:
            l.initialize()


    def feedforward(self, x):
        #print(f"[network] feedforward\nx.shape={x.shape}")
        # we can actually skip the input layer because it does nothing....
        for l in self.mLayers:
            x = l.feedforward(x)

        return x


    def backPropagate(self, dC_over_da):
        for l in reversed(self.mLayers):
            dC_over_da = l.backPropagate(dC_over_da)


    def update(self):
        for l in self.mLayers:
            l.update(self.mLearningRate)


    def train(self, xs, ys):
        """
        :param [in] xs: each row represents an input
        :param [in] ys: each row represents a ground truth output
        """
        print("training")

        indices = np.arange(xs.shape[0])
        for epoch in range(self.mMaxEpoch):
            miniBatchCount = 0
            startTime = time.time()

            np.random.shuffle(indices)
            for i in range(xs.shape[0]):

                yHat = self.feedforward(xs[indices[i], :])

                dC_over_dyHat = self.mCostFunction.getDerivative(ys[indices[i], :], yHat)

                self.backPropagate(dC_over_dyHat)

                if ((1 + i) % self.mBatchSize == 0):
                    self.update()
                    #labels = np.zeros(4)
                    #labels[np.argmax(yHat)] = 1
                    #print(xs[indices[i], :], " --> ", ys[indices[i], :], " --> ", labels)
                    miniBatchCount += 1
                    if (miniBatchCount % 100 == 0):
                        print(f"\tprogress: {miniBatchCount}/{int(len(xs) / self.mBatchSize)}")
            self.update()

            endTime = time.time()

            accuracy = self.validate(xs, ys)

            print(f"==> epoch %d: time = %.2fs, accuracy = %.2f%%" %
                    (epoch, endTime - startTime, accuracy * 100))

            #self.saveToFile("cnn-epoch%d.pkl" % (epoch))



    def validate(self, xs, ys):
        """
        :return: accuracy
        """
        yHats = np.array([self.feedforward(x) for x in xs]).reshape(ys.shape)
        labels = Utils.getPredictions(yHats)
        #for (y, l) in zip(ys, labels):
        #    print(y, " --> ", l)
        accuracy = Utils.compare(labels, ys)

        return accuracy


    def print(self):
        print("[Network] %d layers, %s cost function" % (len(self.mLayers), self.mCostFunction.getName()))
        for l in self.mLayers:
            l.print()


if (__name__ == "__main__"):
    n = Network()

    """
    # a simple sample: finding the pixel with max value in a 2x2 image
    with open("convolutional-network-sample.json", "r") as f:
        jsonStr = f.read()
        n.initialize(jsonStr)

    n.print()

    # input: 4 values between 0 and 1
    # output: which pixel is max
    count = 10000
    xs = np.random.random((count, 4))
    ys = Utils.getPredictions(xs)

    #print("input and output")
    #for (x, y) in zip(xs, ys):
    #    print(x, " --> ", y)

    n.train(xs.reshape(-1, 2, 2), ys)
    """
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

        xs = images.reshape(-1, 28, 28) / 255.0
        ys = np.zeros((len(xs), 10))
        for (i, l) in enumerate(labels):
            ys[i, l] = 1.0


    with open("convolutional-network-mnist.json", "r") as f:
        jsonStr = f.read()
        n.initialize(jsonStr)

    n.print()
    n.train(xs, ys)
