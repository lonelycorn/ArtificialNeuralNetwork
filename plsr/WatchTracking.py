import matplotlib.pyplot as plt
import numpy as np
import os
import  Plsr
import pickle
from pyquaternion import Quaternion

def loadData(folder):
    inputFilename = os.path.join(folder, "result.csv")
    outputFilename = os.path.join(folder, "groundTruth.csv")

    input = np.loadtxt(inputFilename, delimiter=",")

    # remove duplicates
    indices = [i for i in range(1, len(input)) if (input[i-1, 0] < input[i, 0])]
    input = input[indices, :]

    # extract relevant channels
    sampleTime = input[:, 0]
    orientation = input[:, 1:5] # wxyz
    angularVelInDevice =  input[:, 6:9] # xyz
    linearAccInWorld = input[:, 18:21] # xyz

    output = np.loadtxt(outputFilename, delimiter=",")
    ts = output[:, 0] * 1e-9 # ns to s
    position = np.zeros((len(sampleTime), 3))
    for i in range(3):
        position[:, i] = np.interp(sampleTime, ts, output[:, i+1]) # xyz

    return (sampleTime, orientation, angularVelInDevice, linearAccInWorld, position)


def normalize(raw):
    """
    :param [in] raw: each ROW is a sample
    :return (normalized, avg, std)
    """
    avg = np.mean(raw, axis=0)
    std = np.std(raw, axis=0)
    normalized = (raw - avg) / std
    return (normalized, avg, std)


def denormalize(normalized, avg, std):
    raw = normalized * std + avg
    return raw


def assembleInputOutput1(sampleTime, orientation, angularVelInDevice, linearAccInWorld, position):
    """
    :return (X, Y, t) where each ROW of (X, Y) is a pair of sample input / output, and t the
        corresponding sample timestamp
    """
    subsamplingRatio = 10 # pick 1 every so many samples
    windowSize = 50 # number of samples to use

    channelsPerSample = 10 # orientation, angularVelInDevice, linearAccInDevice

    # converting gravity-removed linear acceleration into device ref frame
    # seems to give slightly worse approximation
    # linearAccInWorld RMSE = 0.1166, 0.1042, 0.0846
    # linearAccInDevice RMSE = 0.1592, 0.1247, 0.1339
    linearAccInDevice = np.array(
            [Quaternion(orientation[i, :]).inverse.rotate(linearAccInWorld[i])
                for i in range(len(sampleTime))]) # xyz, gravity-removed

    # horizontally concatenate input channels
    input = np.zeros((len(sampleTime), 10))
    input[:, 0:4] = orientation
    input[:, 4:7] = angularVelInDevice
    input[:, 7:10] = linearAccInWorld #linearAccInDevice

    # all samples in a window
    N = len(sampleTime) - subsamplingRatio * (windowSize - 1)
    X = np.zeros((N, 10 * windowSize))
    # absolute position at the end of the window
    Y = np.zeros((N, 3))
    for i in range(N):
        lastIdx = i + subsamplingRatio * (windowSize - 1)
        X[i, :] = input[i:lastIdx+1:subsamplingRatio, :].flatten()
        Y[i, :] = position[lastIdx, :]

    t = sampleTime[subsamplingRatio * (windowSize - 1):]

    return (X, Y, t)


if (__name__ == "__main__"):
    folder = "watch-tracking"
    pklFilename = "data.pkl"
    resultFilename = "attempt2.csv"

    (sampleTime, orientation, angularVelInDevice, linearAccInWorld, position) = loadData(folder)
    (angularVelInDevice, _, _) = normalize(angularVelInDevice)
    (linearAccInWorld, _, _) = normalize(linearAccInWorld)
    (position, posAvg, posStd) = normalize(position)

    (X, Y, t) = assembleInputOutput1(
            sampleTime, orientation, angularVelInDevice, linearAccInWorld, position)

    with open(os.path.join(folder, pklFilename), "wb") as f:
        pickle.dump((X, Y, t), f)

    """
    with open(os.path.join(folder, pklFilename), "rb") as f:
        (X, Y, t) = pickle.load(f)
    """

    M_star = 10
    (v, p, q, C_YY_history, C_XX_history) = Plsr.decompose(X.T, Y.T, M_star)
    print(f"Input residual covar = {C_XX_history[-1]}")
    print(f"Output residual covar = {C_YY_history[-1]}")

    Y_predicted = Plsr.predict(X.T, v, p, q, M_star)
    Y_predicted = Y_predicted.T

    Y_error = Y - Y_predicted
    rmse = np.sqrt(np.mean(np.multiply(Y_error, Y_error), axis=0))
    print(f"Prediction RMSE = {rmse}")
    print(f"Prediction max error = {np.max(np.abs(Y_error), axis=0)}")

    with open(resultFilename, "w") as f:
        print(f"Saving predicted trajectory to {resultFilename}")
        # t, px, py, pz, qw, qx, qy, qz
        output = np.zeros((len(t), 8))
        output[:, 0] = t * 1e9 # s -> ns
        output[:, 1:4] = denormalize(Y_predicted, posAvg, posStd)
        print(sampleTime.shape)
        print(orientation.shape)
        print(t.shape)
        for i in range(4):
            output[:, 4 + i] = np.interp(t, sampleTime, orientation[:, i])
        np.savetxt(resultFilename, output, fmt="%.4f")



    plt.figure()
    plt.plot(C_YY_history, "o-", label="C_YY")
    plt.plot(C_XX_history, "x-", label="C_XX")
    plt.legend()
    plt.xlabel("iteration")
    plt.ylabel("percentage")
    plt.title("Residual covariance")

    plt.figure()
    for i in range(3):
        plt.subplot(3, 1, i+1)
        plt.plot(t, Y_predicted[:, i], "-", label="prediction")
        plt.plot(t, Y[:, i], ".", label="ground truth")
        plt.ylabel(("X", "Y", "Z")[i])
        plt.legend()
    plt.suptitle(f"prediction using {M_star} components")

    plt.show()


