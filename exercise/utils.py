import numpy as np
from sklearn import model_selection

def normalize(raw):
    """
    :input raw: each row is a sample and each col is a feature channel
    :return normalized samples, per-channel avg, per-channel std
    """
    avg = np.mean(raw, axis=0)
    std = np.std(raw, axis=0)
    normalized = (raw - avg) / std
    return (normalized, avg, std)


def denormalize(normalized, avg, std):
    raw = normalized * std + avg
    return raw


def kernelize(raw):
    (sampleCount, N) = raw.shape

    # augmented features consist of
    #   - raw features (N)
    #   - second-order terms (N + 1) * N / 2
    #   - geometrical features (N - 1) * N / 2 + 1
    M1 = N
    M2 = int((N + 1) * N / 2)
    M3 = int((N - 1) * N / 2 + 1)
    M = M1 + M2 + M3

    output = np.zeros((sampleCount, M))

    output[:, 0 : M1] = raw

    k = M1
    for i in range(N):
        for j in range(i, N):
            output[:, k] = raw[:, i] * raw[:, j]
            k += 1

    for i in range(N):
        for j in range(i + 1, N):
            output[:, k] = np.arctan2(raw[:, i], raw[:, j])
            k += 1
    output[:, k] = np.linalg.norm(raw, axis=1)

    return output


def trainAndTest(X, Y, algorithm, useKernels, normalizeInput):

    testCount = 10
    bingo = np.zeros(testCount, dtype=np.int)

    for i in range(testCount):
        print("[%d/%d] Training using %s samples" % (i + 1, testCount, ["raw", "kernelized"][useKernels]))
        (trainingX, testingX, trainingY, testingY) = model_selection.train_test_split(X, Y, test_size=0.3)
        if (useKernels):
            trainingX = kernelize(trainingX)
            testingX = kernelize(testingX)

        if (normalizeInput):
            (trainingX, _, _) = normalize(trainingX)
            (testingX, _, _) = normalize(testingX)

        predictor = algorithm(trainingX, trainingY)
        predictions = predictor(testingX)

        bingo[i] = 0
        N = len(predictions)
        for (p, t) in zip(predictions, testingY):
            if (p == t):
                bingo[i] += 1
        accuracy = 1.0 * bingo[i] / N
        print(f"\t accuracy = {accuracy * 100 :.2f}% ({bingo[i]}/{N})")

    avg = np.sum(bingo) / (testCount * N) * 100
    std = np.std(bingo / N) * 100
    print(f"Accuracy avg = {avg:.2f}%, std = {std:.2f}%")


if (__name__ == "__main__"):
    pass
