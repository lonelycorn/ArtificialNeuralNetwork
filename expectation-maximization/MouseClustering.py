import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

CLUSTER_COUNT = 3


def generateData():
    """
    :return data. one row [x, y, label] for each sample
    """
    HEAD_COUNT = 390
    LEFT_EAR_COUNT = 100
    RIGHT_EAR_COUNT = 100
    NOISE_COUNT = 10

    data = np.zeros((HEAD_COUNT + LEFT_EAR_COUNT + RIGHT_EAR_COUNT + NOISE_COUNT, 3))
    idx = 0

    data[idx : idx + HEAD_COUNT, :2] = \
            np.array([0.5, 0.5]) + np.random.randn(HEAD_COUNT, 2) * np.array([0.1, 0.1])
    data[idx : idx + HEAD_COUNT, -1] = 0
    idx += HEAD_COUNT

    data[idx : idx + LEFT_EAR_COUNT, :2] = \
            np.array([0.25, 0.75]) + np.random.randn(LEFT_EAR_COUNT, 2) * np.array([0.05, 0.05])
    data[idx : idx + LEFT_EAR_COUNT, -1] = 1
    idx += LEFT_EAR_COUNT

    data[idx : idx + RIGHT_EAR_COUNT, :2] = \
            np.array([0.75, 0.75]) + np.random.randn(RIGHT_EAR_COUNT, 2) * np.array([0.05, 0.05])
    data[idx : idx + RIGHT_EAR_COUNT, -1] = 2
    idx += RIGHT_EAR_COUNT

    data[idx : idx + NOISE_COUNT, :2] = noise = np.random.rand(NOISE_COUNT, 2)
    data[idx : idx + NOISE_COUNT, -1] = 3
    idx += NOISE_COUNT

    return data


def visualizeData(points, labels, title):
    fig = plt.figure()

    idx = (labels == 0)
    plt.plot(points[idx, 0], points[idx, 1], "r.")

    idx = (labels == 1)
    plt.plot(points[idx, 0], points[idx, 1], "g.")

    idx = (labels == 2)
    plt.plot(points[idx, 0], points[idx, 1], "b.")

    idx = (labels > 2)
    if (np.sum(idx) > 0):
        plt.plot(points[idx, 0], points[idx, 1], "k.")
        plt.legend(["head", "left ear", "right ear", "noise"])

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title(title)

    return fig


def kMeansClustering(points, K):
    """
    :return (centers, labels)
    """
    solver = KMeans(n_clusters=K, verbose=0)
    solver.fit(points)

    return (solver.cluster_centers_, solver.labels_)


def gaussianMixtureModelClustering(points, K):
    """
    :return (centers, labels)
    """
    MAX_ITERATION_COUNT = 15
    PLOT_ITERATION = False

    def getProbability(x, mu, var):
        # 2D Gaussian
        denom = 2 * np.pi * np.sqrt(np.linalg.det(var))
        delta = x - mu # 1x2
        exponent = np.matmul(delta, np.matmul(np.linalg.inv(var), delta.T))
        return 1.0 / denom * np.exp(-0.5 * exponent)

    def computeWeightedProbability(X, mu, var, pi):
        # wp[i, j] = pi[j] * getProbability(x[i], mu[j], var[j])
        N = len(X)
        K = len(mu)
        wp = np.zeros((N, K))
        for j in range(K):
            wp[:, j] = np.array([pi[j] * getProbability(X[i, :], mu[j, :], var[j, :, :]) for i in range(N)])
        return wp

    def visualizeClusters(mu, var, colors):
        fig = plt.figure(0)
        plt.clf()
        ax = fig.add_subplot(111, aspect="equal")

        # sample points
        ax.plot(points[:, 0], points[:, 1], ".")

        # confidence ellipses
        for j in range(0, K):
            (U, S, VT) = np.linalg.svd(var[j, :, :])
            theta = np.arctan2(U[0, 1], U[0, 0])
            for k in range(1, 4):
                el = mpl.patches.Ellipse(
                    xy=mu[j, :],
                    width=np.sqrt(S[0]) * k * 2,
                    height=np.sqrt(S[1]) * k * 2,
                    angle = np.rad2deg(theta))
                ax.add_artist(el)
                el.set_alpha(0.6 - k * 0.1)
                el.set_facecolor(colors[j])
        plt.title(f"iteration {iteration}")
        fig.canvas.draw()
        plt.pause(0.2)


    N = len(points)

    # initialize centers with K-Means
    (mu, labels) = kMeansClustering(points, K)
    var = np.zeros((K, 2, 2))
    pi = np.zeros(K)
    for j in range(K):
        indices = (labels == j)
        count = np.sum(indices)
        delta = points[indices, :] - mu[j, :]
        var[j, :, :] = np.matmul(delta.T, delta) / (count - 1) # unbiased estimate
        pi[j] = count / N
    gamma = np.zeros((N, K)) # to be filled later

    if (PLOT_ITERATION):
        colors = np.random.rand(K, 3)

    #print(f"points\n{points}")
    #print("initilizing with K-Means")
    #print(f"labels=\n{labels}")
    #print(f"mu =\n{mu}")
    #print(f"var =\n{var}")
    #print(f"pi=\n{pi}")

    # E-M iterations
    for iteration in range(MAX_ITERATION_COUNT):

        mu1 = np.zeros(mu.shape)
        var1 = np.zeros(var.shape)
        pi1 = np.zeros(pi.shape)
        if (PLOT_ITERATION):
            visualizeClusters(mu, var, colors)

        weightedProbability = computeWeightedProbability(points, mu, var, pi)
        sumWeightedProbability = np.sum(weightedProbability, axis=1)

        for j in range(K):
            gamma = weightedProbability[:, j] / sumWeightedProbability

            sumGamma = np.sum(gamma)

            # distribution parameters that maximizes log likelihood
            mu1[j, :] = np.sum((gamma.T * points.T).T, axis=0) / sumGamma
            delta = points - mu1[j, :] # (N, 2)
            var1[j, :, :] = np.matmul(delta.T, np.matmul(np.diag(gamma), delta)) / sumGamma
            pi1[j] = sumGamma / N


        mu = mu1
        var = var1
        pi = pi1

        #print(f"iteration {iteration}")
        #print(f"weightedProbability =\n{weightedProbability}")
        #print(f"mu =\n{mu}")
        #print(f"var =\n{var}")
        #print(f"pi=\n{pi}")

    labels = np.argmax(weightedProbability, axis=1)

    return (mu, labels)


if (__name__ == "__main__"):
    data = generateData()

    points = data[:, :2]
    labels = data[:, 2]

    visualizeData(points, labels, "ground truth")

    (kMeansCenters, kMeansLabels) = kMeansClustering(points, CLUSTER_COUNT)
    print("K Means cluster centers:")
    for c in kMeansCenters:
        print(f"\t{c}")
    visualizeData(points, kMeansLabels, "K Means")

    (gmmCenters, gmmLabels) = gaussianMixtureModelClustering(points, CLUSTER_COUNT)
    print("Gaussian Mixture Model centers:")
    for c in gmmCenters:
        print(f"\t{c}")
    visualizeData(points, gmmLabels, "Gaussian Mixture Model")

    plt.show()

