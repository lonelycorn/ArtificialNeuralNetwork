import numpy as np
import ActivationFunction
import CostFunction
import json

class Node:
    """
    Each node represents a neuron / perceptron
    """
    def __init__(self, id, activationFunction, tag=None):
        self.mId = id
        if (tag is None):
            self.mTag = "hidden"
        else:
            self.mTag = tag

        # neuron definition
        self.mNodeIdToInputIdx = {}
        self.mInputs = []
        self.mOutputs = []
        self.mActivationFunction = activationFunction

        # neuron parameters
        self.mInputWeights = []
        self.mInputBias = 0.0

        # latest input activations to the neuron
        #       [ ... a_j ...]
        self.mLatestInputs = []
        # latest output activation of the neuron
        #       z = sum( w_j * a_j) + b
        #       a = f(z) where f() is the activation function
        self.mLatestActivation = 0
        #       d_a_over_d_z
        self.mLatestActivationDerivative = 0
        # latest partial derivative of the cost to z
        #       par_C_over_par_z
        self.mLatestCostPartialDerivative = 0

        # for (stochastic) gradient descent
        self.mDeltaInputWeights = []
        self.mDeltaInputBias = 0.0
        self.mSampleCount = 0


    @property
    def id(self):
        return self.mId


    @property
    def tag(self):
        return self.mTag


    def getLatestActivation(self):
        return self.mLatestActivation


    def getLatestCostPartialDerivative(self):
        return self.mLatestCostPartialDerivative


    def getInputWeight(self, neuronId):
        return self.mInputWeights[self.mNodeIdToInputIdx[neuronId]]


    def registerInput(self, n):
        """
        use during configuration
        """
        if (n not in self.mInputs):
            self.mNodeIdToInputIdx[n.id] = len(self.mInputs)
            self.mInputs.append(n)


    def registerOutput(self, n):
        """
        use during configuration
        """
        if (n not in self.mOutputs):
            self.mOutputs.append(n)


    def initialize(self):
        inputCount = len(self.mInputs)
        # TODO: compare with He initialization
        weightStdDev = np.sqrt(1.0 / inputCount)
        self.mInputWeights = np.random.normal(0.0, weightStdDev, (inputCount, 1))
        self.mInputBias = np.random.normal(0.0, 1.0)

        self.mDeltaInputWeights = np.zeros(inputCount)
        self.mDeltaInputBias = 0.0
        self.mSampleCount = 0

    def feedforward(self):
        """
        :return activation of the neuron
        """
        self.mLatestInputs = np.array([n.getLatestActivation() for n in self.mInputs])
        z = np.dot(self.mLatestInputs, self.mInputWeights) + self.mInputBias
        self.mLatestActivation = self.mActivationFunction.getValue(z)
        self.mLatestActivationDerivative = self.mActivationFunction.getDerivative(z)
        return self.mLatestActivation


    def backPropagate(self):
        """
        :return cost partial derivateive
        """
        par_C_over_par_a = \
                np.sum([n.getLatestCostPartialDerivative() * n.getInputWeight(self.mId) for n in self.mOutputs])

        # par_C_over_par_z = par_C_over_par_a * d_a_over_d_z
        self.mLatestCostPartialDerivative = par_C_over_par_a * self.mLatestActivationDerivative

        self.mDeltaInputWeights += self.mLatestCostPartialDerivative * self.mLatestInput
        self.mDeltaInputBias += self.mLatestCostPartialDerivative
        self.mSampleCount += 1
        return self.mLatestCostPartialDerivative


    def update(self, learningRate):
        """
        update parameters using gradient descent
        """
        k = learningRate / mSampleCount
        self.mInputWeights -= k * self.mDeltaInputWeights
        self.mInputBias -= k * self.mDeltaInputBias

        # prepare for the next update
        self.mDeltaInputWeights = np.zeros(len(self.mInputWeights))
        self.mDeltaInputBias = 0.0
        self.mSampleCount = 0


    def print(self):
        print(f"[neuron] id = {self.mId}, tag = {self.mTag}")
        print(f"    {len(self.mInputs)} inputs: {[n.id for n in self.mInputs]}")
        print(f"    {len(self.mOutputs)} outputs:  {[n.id for n in self.mOutputs]}")


class InputNode(Node):
    def __init__(self, id):
        Node.__init__(self, id, ActivationFunction.Linear, "input")

    def initialize(self):
        pass

    def feedforward(self):
        self.mLatestActivation = x
        return self.mLatestActivation

    def backPropagate(self):
        raise RuntimeError("An InputNode does not backPropagate")

    def update(self, learningRate):
        raise RuntimeError("An InputNode does not update")

class OutputNode(Node):
    def __init__(self, id, activationFunction):
        Node.__init__(self, id, ActivationFunction, "output")

    def backPropagate(self, costPartialDerivative):
        self.mLatestCostPartialDerivative = costPartialDerivative

        self.mDeltaInputWeights += self.mLatestCostPartialDerivative * self.mLatestInput
        self.mDeltaInputBias += self.mLatestCostPartialDerivative
        self.mSampleCount += 1
        return self.mLatestCostPartialDerivative

class Graph:
    def __init__(self):
        # only indices
        self.mInputNodeIndices = []
        self.mOutputNodeIndices = []
        # all node instances, sorted topologically
        self.mNodes = []
        self.mCostFunction = None

        self.mBatchSize = 1
        self.mLearningRate = 0.1

    def saveToFile(self, filename):
        pass

    def loadFromFile(self, filename):
        pass

    def initialize(self, jsonStr):
        """
        JSON format:
        {
            "graph": {
                "nodes": array of Node
                "edges": array of Edge,
                "cost_function": str
            },
            "hyperparameters": {
                "learning_rate": float
                "batch_size": int
            }
        }

        where:
        Node:
        {
            "id": int ( must be unique )
            "activation_function": str ( "sigmoid", "tanh", "relu", "softplus", "linear" )
            "tag": str ( "input", "output", "hidden" )
        }

        Edge:
            [ int, int ]
        ID of the from source node and estination node, respectively

        cost_function: one of ( "quadratic", "cross_entropy" )

        """
        root = json.loads(jsonStr)
        if (not isinstance(root, dict)):
            raise InputError("Invalid JSON format")
        graph = root["graph"]
        hyperparameters = root["hyperparameters"]

        # parse graph
        nodes = {}
        for n in graph["nodes"]:
            id = int(n["id"])
            if (id in nodes):
                raise InputError(f"Duplicated node id {id}")

            af = n["activation_function"]
            tag = n["tag"]
            if ("input" == tag):
                nodes.update({id: InputNode(id)})
                self.mInputNodeIndices.append(id)
            elif ("output" == tag):
                nodes.update({id: OutputNode(id, ActivationFunction.create(af))})
                self.mOutputNodeIndices.append(id)
            elif ("hidden" == tag):
                nodes.update({id: Node(id, ActivationFunction.create(af), tag)})
            else:
                raise InputError(f"Unsupported node tag {tag}")

        edges = graph["edges"]
        indegrees = [set() for i in range(len(nodes))]
        for e in edges:
            src = e[0]
            dst = e[1]
            nodes[src].registerOutput(nodes[dst])
            nodes[dst].registerInput(nodes[src])

            indegrees[dst].add(src)


        self.mCostFunction = CostFunction.create(graph["cost_function"])

        # parse hyperparameters
        self.mLearningRate = float(hyperparameters["learning_rate"])
        self.mBatchSize = int(hyperparameters["batch_size"])

        # topological sort
        nodeCount = len(nodes)
        processed = [False for i in range(nodeCount)]
        self.mNodes.clear()

        for i in range(nodeCount):
            id = None
            for j in range(nodeCount):
                if ((not processed[j]) and (len(indegrees[j]) == 0)):
                    id = j
                    break
            if (id is None):
                raise RuntimeError("Not a valid graph")

            processed[id] = True
            self.mNodes.append(nodes[id])
            for j in range(nodeCount):
                indegrees[j].discard(id)

    def feedforward(self, x):
        """
        :return yHat
        """
        pass

    def train(self, x, y):
        pass

    def print(self):
        print(f"[graph] %d nodes (%d input, %d output), %s cost function" % (
            len(self.mNodes),
            len(self.mInputNodeIndices),
            len(self.mOutputNodeIndices),
            self.mCostFunction.getName()))

        for n in self.mNodes:
            n.print()

    def doTopologicalSort():
        pass


if (__name__ == "__main__"):
    g = Graph()
    with open("node-graph-sample.json", "r") as f:
        jsonStr = f.read()
        g.initialize(jsonStr)

    g.print()

