import numpy as np


import Utils


def extractPrincipalComponents(image, blockSize):
    """
    :param [in] img:    2D (grayscale) or 3D (rows, cols, channels) numpy array
    :param [in] blockSize: 2-tuple; the shape of each block
    :return (importance, newBases, newCoords, dimensionAvg, dimensionStd)
    """
    if (len(image.shape) == 2):
        image = image.reshape((image.shape[0], image.shape[1], 1))

    (imageRows, imageCols, imageChns) = image.shape
    (blockRows, blockCols) = blockSize

    # sanity check
    if ((imageRows <= blockRows) or (imageCols <= blockCols) or
        (imageRows % blockRows > 0) or (imageCols % blockCols > 0)):
        raise ValueError("Improper image size / block size")

    # each sample is a flattened block (blockRows, blockCols, imageChns)
    sampleCount = int(imageRows / blockRows) * int(imageCols / blockCols) * 1
    sampleLength = blockRows * blockCols * imageChns

    # each row is a raw sample
    raw = np.zeros((sampleCount, sampleLength))
    N = 0
    for r in range(0, imageRows, blockRows):
        for c in range(0, imageCols, blockCols):
            raw[N, :] = image[r : r + blockRows, c : c + blockCols, :].flatten()
            N += 1

    # normalization by dimension
    dimensionAvg = np.mean(raw, axis=0)
    dimensionStd = np.std(raw, axis=0)

    # NOTE: numpy broadcasting only works if the last dimension matches
    samples = (raw - dimensionAvg) / dimensionStd

    (importance, newBases, newCoords) = Utils.pca(samples)

    # for visualizing the principal
    #from matplotlib import pyplot
    #pyplot.figure()
    #pyplot.imshow(newBases[:, 0].reshape(blockSize), cmap="gray")
    #pyplot.title("first principal component")
    #pyplot.figure()
    #pyplot.imshow(newBases[:, 1].reshape(blockSize), cmap="gray")
    #pyplot.title("second principal component")
    #pyplot.figure()
    #pyplot.imshow(newBases[:, 2].reshape(blockSize), cmap="gray")
    #pyplot.title("thrid principal component")

    return (importance, newBases, newCoords, dimensionAvg, dimensionStd)



def constructFromPrincipalComponents(pcCount, newBases, newCoords, dimensionAvg, dimensionStd, imageShape, blockSize):
    """
    :return a numpy array of imageShape; the reconstructed image
    """
    if (len(imageShape) == 2):
        (imageRows, imageCols) = imageShape
        imageChns = 1
    else:
        (imageRows, imageCols, imageChns) = imageShape
    (blockRows, blockCols) = blockSize

    # sanity check
    if ((imageRows <= blockRows) or (imageCols <= blockCols) or
        (imageRows % blockRows > 0) or (imageCols % blockCols > 0)):
        raise ValueError("Improper image size / block size")

    # reduced-dimension bases
    samples = np.matmul(newCoords[:, 0 : pcCount + 1], newBases[:, 0 : pcCount + 1].T)

    # de-normalize
    samples = samples * dimensionStd + dimensionAvg

    image = np.zeros((imageRows, imageCols, imageChns))
    N = 0
    for r in range(0, imageRows, blockRows):
        for c in range(0, imageCols, blockCols):
            image[r : r + blockRows, c : c + blockCols, :] = samples[N, :].reshape((blockRows, blockCols, imageChns))
            N += 1

    if (imageChns == 1):
        image = image.reshape((imageRows, imageCols))

    return image


def compressImage(image, targetInformationRatio, blockSize=(8, 8)):
    """
    :return (compressedImage, diffImage, pcCount, informationRatio)
    """
    if ((targetInformationRatio <= 0) or (targetInformationRatio > 1.0)):
        raise ValueError("Unrealistic target information ratio")

    (importance, newBases, newCoords, dimensionAvg, dimensionStd) = \
            extractPrincipalComponents(image, blockSize)
    pcCount = 1
    sumImportance = np.sum(importance)
    informationRatio = importance[0] / sumImportance
    while (informationRatio < targetInformationRatio):
        informationRatio += importance[pcCount] / sumImportance
        pcCount += 1
    print(f"Using {pcCount} principal components")
    print(f"Information ratio = {informationRatio}")

    # 1 byte for each channel at every pixel
    rawSize = np.prod(image.shape)
    # for each sample: pcCount bytes for new coord;
    # for each principle component: baseLength bytes
    # for each dimension: 2 bytes for avg and std
    (baseLength, baseCount) = newBases.shape
    (sampleCount, _) = newCoords.shape
    compressedSize = pcCount * baseLength + pcCount * sampleCount + 2 * baseCount
    print(f"Size compresson ratio = {compressedSize * 1.0 / rawSize} ({compressedSize}/{rawSize})")


    compressedImage = constructFromPrincipalComponents(
            pcCount,
            newBases,
            newCoords,
            dimensionAvg,
            dimensionStd,
            image.shape,
            blockSize)

    diffImage = image - compressedImage

    return (compressedImage, diffImage, pcCount, informationRatio)


if (__name__ == "__main__"):
    from matplotlib import pyplot, image

    tryColorImage = True
    targetInformationRatio = 0.95
    blockSize = (8, 8)

    if (tryColorImage):
        colorImage = image.imread("lena.png")
        (compressedColorImage, diffColorImage, pcCount, informationRatio) = compressImage(
                colorImage, targetInformationRatio, blockSize)

        # show original and compressed images
        pyplot.figure()
        pyplot.imshow(colorImage)
        pyplot.title("original color")

        description = f"{pcCount}/{np.prod(blockSize)} components"
        pyplot.figure()
        pyplot.imshow(np.abs(compressedColorImage)) # FIXME: some values are outside [0, 1]
        pyplot.title(f"compressed color ({description})")

        pyplot.figure()
        pyplot.imshow(np.abs(diffColorImage)) # FIXME: some values are outside [0, 1]
        pyplot.title(f"diff color ({description})")

    else:
        grayImage = Utils.loadGrayscaleImage("lena.png")
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


        # show original and compressed images
        pyplot.figure()
        pyplot.imshow(grayImage, cmap="gray")
        pyplot.title("original grayscale")

        description = f"{pcCount}/{np.prod(blockSize)} components"
        pyplot.figure()
        pyplot.imshow(compressedGrayImage, cmap="gray")
        pyplot.title(f"compressed grayscale ({description})")

        pyplot.figure()
        pyplot.imshow(diffGrayImage, cmap="gray")
        pyplot.title(f"diff grayscale ({description})")

    pyplot.show()
