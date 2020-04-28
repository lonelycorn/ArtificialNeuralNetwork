import colorsys
import numpy as np
from matplotlib import image

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

    # Singular-value decomposition gives the eigenvalues (S), left-eigenvectors (U)
    # and right-eigenvectors (V)
    # since we put newBases on the right hand side of newCoords, we need the right-
    # eigenvectors
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

    # A eigenvalue represents the covar explained by this principal component
    # more ability to explain covar --> higher importance
    importance = S
    newBases = V
    # newCoords * newBases == samples == oldCoords * oldBases
    # Note oldBases == identity matrix, and oldCoords == samples
    newCoords = np.matmul(samples, VT)

    return (importance, newBases, newCoords)

def loadGrayscaleImage(filename):
    img = image.imread(filename)

    # color image: (rows, cols, channels)
    if (len(img.shape) > 2):
        # convert to grayscale ("Y"/luma in YIQ)
        img = colorsys.rgb_to_yiq(r=img[:, :, 0], g=img[:, :, 1], b=img[:, :, 2])[0]
    return img
