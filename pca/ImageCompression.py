import numpy as np


def pca(samples):
    """
    :param [in] samples: N-by-M matrix; each row is a M-dimensional sample
    :return (importance, newBases, newCoords)
        -   importance: M array; covar / importance of each new basis
        -   newBases: M-by-M matrix; each col is a new basis
        -   newCoords: N-by-M matrix; each row is the new coord of the original sample
                expressed with the new bases
        NOTE: the original samples could be reconstructed as
                samples = newCoords * newBases
    """
    covar = np.matmul(samples.T, samples)

    U, S, VT = np.linalg.svd(covar)
    V = VT.T # right-eigenvectors

    #print("covar")
    #print(covar)
    #print("S")
    #print(S)
    #print("covar * V[:, 0]")
    #print(np.matmul(covar, V[:, 0]))
    #print("S[0] * V[:, 0]")
    #print(S[0] * V[:, 0])

    importance = S
    newBases = V
    newCoords = np.matmul(samples, VT)

    return (importance, newBases, newCoords)



def extractPrincipalComponents(image, blockSize):
    """
    :param [in] img:    2D numpy array; the grayscale image
    :param [in] blockSize: 2-tuple; the shape of each block
    """
    (imageRows, imageCols) = image.shape
    (blockRows, blockCols) = blockSize

    # sanity check
    if ((imageRows <= blockRows) or (imageCols <= blockCols) or
        (imageRows % blockRows > 0) or (imageCols % blockCols > 0)):
        raise ValueError("Improper image size / block size")

    # each sample is a flattened block
    sampleCount = int(imageRows / blockRows) * int(imageCols / blockCols)
    sampleLength = blockRows * blockCols

    # each row is a raw sample
    raw = np.zeros((sampleCount, sampleLength))
    N = 0
    for r in range(0, imageRows, blockRows):
        for c in range(0, imageCols, blockCols):
            raw[N, :] = image[r : r + blockRows, c : c + blockCols].flatten()
            N += 1

    # normalization
    avg = np.mean(raw, axis=1) # mean of each sample
    std = np.std(raw, axis=1) # std dev of each sample

    # NOTE: numpy broadcasting only works if the last dimension matches
    samples = ((raw.T - avg.T) / std.T).T

    #print(f"raw\n{raw}")
    #print(f"avg\n{avg}")
    #print(f"std\n{std}")
    #print(f"samples\n{samples}")

    (importance, newBases, newCoords) = pca(samples)

    return (importance, newBases, newCoords, avg, std)


def constructFromPrincipalComponents(pcCount, newBases, newCoords, blockAvg, blockStd, imageSize, blockSize):
    """
    :return a 2D numpy array; the reconstructed image
    """
    (imageRows, imageCols) = imageSize
    (blockRows, blockCols) = blockSize

    # sanity check
    if ((imageRows <= blockRows) or (imageCols <= blockCols) or
        (imageRows % blockRows > 0) or (imageCols % blockCols > 0)):
        raise ValueError("Improper image size / block size")

    samples = np.matmul(newCoords[:, 0 : pcCount], newBases[0 : pcCount, :])
    #print(f"recovered normalized samples.shape = {samples.shape}")

    samples = (samples.T * blockStd.T + blockAvg.T).T
    #print(f"recovered raw samples.shape = {samples.shape}")

    image = np.zeros(imageSize)
    N = 0
    for r in range(0, imageRows, blockRows):
        for c in range(0, imageCols, blockCols):
            image[r : r + blockRows, c : c + blockCols] = samples[N, :].reshape(blockSize)
            N += 1

    return image


def compressImage(image, targetInformationRatio, blockSize=(8, 8)):
    """
    :return (compressedImage, diffImage, pcCount, informationRatio)
    """
    if ((targetInformationRatio <= 0) or (targetInformationRatio > 1.0)):
        raise ValueError("Unrealistic target information ratio")

    (importance, newBases, newCoords, blockAvg, blockStd) = \
            extractPrincipalComponents(image, blockSize)
    pcCount = 1
    sumImportance = np.sum(importance)
    informationRatio = importance[0] / sumImportance
    while (informationRatio < targetInformationRatio):
        informationRatio += importance[pcCount] / sumImportance
        pcCount += 1
    print(f"Using {pcCount} principal components")
    print(f"Information ratio = {informationRatio}")

    # 1 byte for each pixel
    rawSize = np.prod(image.shape)
    # for each sample: pcCount bytes for new coord, 2 bytes for avg and std
    (baseLength, _) = newBases.shape
    (sampleCount, _) = newCoords.shape
    compressedSize = pcCount * baseLength + (pcCount + 2 ) * sampleCount
    print(f"Size compresson ratio = {compressedSize * 1.0 / rawSize} ({compressedSize} / {rawSize})")


    compressedImage = constructFromPrincipalComponents(
            pcCount,
            newBases,
            newCoords,
            blockAvg,
            blockStd,
            image.shape,
            blockSize)

    diffImage = image - compressedImage

    return (compressedImage, diffImage, pcCount, informationRatio)



if (__name__ == "__main__"):
    from matplotlib import image, pyplot
    import colorsys

    # color image: (rows, cols, channels)
    colorImage = image.imread("lena.png")
    #print(type(colorImage))
    #print(colorImage.shape)

    # convert to grayscale ("Y"/luma in YIQ)
    grayImage = colorsys.rgb_to_yiq(
            r=colorImage[:, :, 0],
            g=colorImage[:, :, 1],
            b=colorImage[:, :, 2])[0]
    #print(type(grayImage))
    #print(grayImage.shape)

    targetInformationRatio = 0.95
    blockSize = (8, 8)

    (compressedGrayImage, diffGrayImage, pcCount, informationRatio) = compressImage(
            grayImage, targetInformationRatio, blockSize)


    # show importance of each principal components
    #(importance, _, _, _, _) = extractPrincipalComponents(grayImage, blockSize)
    #pyplot.figure()
    #pyplot.plot(importance / np.sum(importance) * 100, "rx-")
    #pyplot.plot(np.cumsum(importance) / np.sum(importance) * 100, "k^-")
    #pyplot.legend(["individual", "cumulative"])
    #pyplot.title("Importance of principal components")
    #pyplot.xlabel("principle component ID")
    #pyplot.ylabel("importance (%)")


    # show original images
    #pyplot.figure()
    #pyplot.imshow(colorImage)
    #pyplot.title("original color")

    pyplot.figure()
    pyplot.imshow(grayImage, cmap="gray")
    pyplot.title("original grayscale")

    pyplot.figure()
    pyplot.imshow(compressedGrayImage, cmap="gray")
    pyplot.title(f"compressed grayscale ({pcCount} principal components)")

    pyplot.figure()
    pyplot.imshow(diffGrayImage, cmap="gray")
    pyplot.title(f"diff grayscale ({pcCount} principal components)")

    pyplot.show()
