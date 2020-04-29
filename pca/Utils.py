import colorsys
import numpy as np
from matplotlib import image

def pca(samples):
    """
    :param [in] samples: N-by-M matrix; each row is a M-dimensional sample
        NOTE: samples should be normalized: for every dimension, avg = 0 and std = 1
    :return (importance, newBases, newCoords)
        -   importance: M array; covar / importance of each new basis
        -   newBases: M-by-M matrix; each col is a new basis
        -   newCoords: N-by-M matrix; each row is the new coord of the original sample
                expressed with the new bases. A.k.a. score; loading
        NOTE: the original samples could be reconstructed as
                samples = newCoords * newBases.T
    """
    (sampleCount, coordLength) = samples.shape
    # NOTE: using "sample covar" here (divided by n-1 instead of n)
    covar = np.matmul(samples.T, samples).astype(np.double) / (sampleCount - 1)

    # Singular-value decomposition gives the eigenvalues (S), left-eigenvectors (U)
    # and right-eigenvectors (V)
    # since we put newBases on the right hand side of newCoords, we need the right-
    # eigenvectors
    U, S, VT = np.linalg.svd(covar)
    # col of V are principal directions
    V = VT.T

    # A eigenvalue represents the covar explained by this principal component
    # more ability to explain covar --> higher importance
    importance = S
    newBases = V
    newCoords = np.matmul(samples, V)

    return (importance, newBases, newCoords)

def loadGrayscaleImage(filename):
    img = image.imread(filename)

    # color image: (rows, cols, channels)
    if (len(img.shape) > 2):
        print("converting to grayscale")
        # convert to grayscale ("Y"/luma in YIQ)
        img = colorsys.rgb_to_yiq(r=img[:, :, 0], g=img[:, :, 1], b=img[:, :, 2])[0]
    return img
