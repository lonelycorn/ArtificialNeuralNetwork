import matplotlib.pyplot as plt
import numpy as np
import os
import  Plsr
import pickle
from pyquaternion import Quaternion

ACCEL_RANGE = 8 * 9.80665 # m/s^2
GYRO_RANGE = 2000 / 180.0 * 3.1415926 # rad/s

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

    # NOTE: converting gravity-removed linear acceleration into device ref frame
    # seems to give slightly worse approximation
    #   * linearAccInWorld: pos RMSE = 0.0105, 0.0219. 0.0241
    #   * linearAccInDevice: pos RMSE = 0.0106, 0.0219, 0.0242
    #linearAccInDevice = np.array(
    #        [Quaternion(orientation[i, :]).inverse.rotate(linearAccInWorld[i])
    #            for i in range(len(sampleTime))]) # xyz, gravity-removed

    # horizontally concatenate input channels
    input = np.zeros((len(sampleTime), 10))
    input[:, 0:4] = orientation
    input[:, 4:7] = angularVelInDevice
    input[:, 7:10] = linearAccInWorld

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


def prepareInputOutput(folder, pklFilename):
    (sampleTime, orientation, angularVelInDevice, linearAccInWorld, position) = loadData(folder)

    # NOTE: we're not "normalizing" angularVelInDevice and linearAccInWorld because
    # that would remove sensor bias, which is not realistic. Instead we just "scale"
    # it so that the corresponding ranges are [-1.0, 1.0]
    angularVelInDevice = angularVelInDevice / ACCEL_RANGE
    linearAccInWorld = linearAccInWorld / GYRO_RANGE
    # FIXME: we should probably just "scale" position similar to above
    (position, posAvg, posStd) = normalize(position)

    (X, Y, t) = assembleInputOutput1(
            sampleTime, orientation, angularVelInDevice, linearAccInWorld, position)

    with open(os.path.join(folder, pklFilename), "wb") as f:
        pickle.dump((X, Y, t), f)

    """
    with open(os.path.join(folder, pklFilename), "rb") as f:
        (X, Y, t) = pickle.load(f)
    """

    return (X, Y, t, sampleTime, orientation, position, posAvg, posStd)


if (__name__ == "__main__"):
    trainingFolder = "watch-simulation-bias-0_02" # bias in X
    testingFolder = "watch-simulation-bias-0_1" # bias in Z

    M_star = 10
    pklFilename = "data.pkl"
    resultFilename = "attempt1.csv"

    # Training
    print(f"Training with {M_star} components using data in {trainingFolder}")
    (X, Y, t, sampleTime, orientation, position, posAvg, posStd) = prepareInputOutput(trainingFolder, pklFilename)
    (v, p, q, C_YY_history, C_XX_history) = Plsr.decompose(X.T, Y.T, M_star)
    print(f"Input residual covar = {C_XX_history[-1]}")
    print(f"Output residual covar = {C_YY_history[-1]}")

    # Testing
    print(f"Testing with {M_star} components using data in {testingFolder}")
    (X, Y, t, sampleTime, orientation, position, posAvgTesting, posStdTesting) = \
            prepareInputOutput(testingFolder, pklFilename)

    Y_predicted = Plsr.predict(X.T, v, p, q, M_star)
    Y_predicted = Y_predicted.T

    # Recover trajectories
    predictedPos = denormalize(Y_predicted, posAvg, posStd)
    originalPos = denormalize(position, posAvgTesting, posStdTesting)
    interpolatedOriginalPos = np.zeros(predictedPos.shape)
    for i in range(3):
        interpolatedOriginalPos[:, i] = np.interp(t, sampleTime, originalPos[:, i]) # xyz

    # Compute simple metrics
    posError = interpolatedOriginalPos - predictedPos
    rmse = np.sqrt(np.mean(np.multiply(posError, posError), axis=0))
    print(f"Prediction RMSE = {rmse}")
    print(f"Prediction max error = {np.max(np.abs(posError), axis=0)}")

    # Output predicted trajectory
    with open(resultFilename, "w") as f:
        print(f"Saving predicted trajectory to {resultFilename}")
        # t, px, py, pz, qw, qx, qy, qz
        output = np.zeros((len(t), 8))
        output[:, 0] = t * 1e9 # s -> ns
        output[:, 1:4] = predictedPos
        for i in range(4):
            output[:, 4 + i] = np.interp(t, sampleTime, orientation[:, i])
        np.savetxt(resultFilename, output, fmt="%.4f")

    # Plotting
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
        plt.plot(sampleTime, originalPos[:, i], "-", label="gt")
        plt.plot(t, predictedPos[:, i], "-", label="prediction")
        plt.ylabel(("X", "Y", "Z")[i])
        plt.legend()
    plt.suptitle(f"Predicted trajectory using {M_star} components")

    plt.show()

