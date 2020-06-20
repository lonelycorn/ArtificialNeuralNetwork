import numpy as np
import copy


def decompose(X_raw, Y_raw, M_star):
    """
    :param [in] X_raw: M-by-N matrix where each COL is a sample input
    :param [in] Y_raw: L-by-N matrix where each COL is a sample output
    :return (v, p, q, C_YY_history, C_XX_history)
    """
    assert(X_raw.shape[1] == Y_raw.shape[1])

    # make a copy so we don't introduce surprises
    X = copy.deepcopy(X_raw)
    Y = copy.deepcopy(Y_raw)

    # record the dimensions
    (M, N) = X.shape
    (L, _) = Y.shape

    # L-by-M
    C_YX = np.matmul(Y, X.T) / (N - 1)
    # M-by-M
    C_XX = np.matmul(X, X.T) / (N - 1)
    # L-by-L
    C_YY = np.matmul(Y, Y.T) / (N - 1)

    # each col is a new basis of the input
    v = np.zeros((M, M_star))
    # loading for output
    q = np.zeros((L, M_star))
    # loading for input
    p = np.zeros((M, M_star))

    # residual information after using first [0..k] latent variables
    C_YY_history = np.zeros(M_star)
    C_XX_history = np.zeros(M_star)
    C_YY_init_trace = np.trace(C_YY)
    C_XX_init_trace = np.trace(C_XX)
    #print(f"initial tr(C_YY) = {C_YY_init_trace}")
    #print(f"initial tr(C_XX) = {C_XX_init_trace}")

    # deflated samples
    # NOTE: we're not making copies because PLSR only needs C_XX and C_YX
    X1 = X
    Y1 = Y
    for i in range(M_star):
        (W, D, VT) = np.linalg.svd(C_YX)

        # save the most dominant direction
        # M-by-1
        v0 = VT[0, :]
        v[:, i] = v0
        v0 = v0.reshape(-1, 1)

        # 1-by-N, input score
        z0 = np.matmul(v0.T, X).reshape(1, -1)

        # 1-by-1, common denominator
        D = np.matmul(v0.T, np.matmul(C_XX, v0))

        # L-by-1, output loading
        # this is the direction that leads to best prediction of Y using z
        q0 = np.matmul(C_YX, v0) / D
        q[:, i] = q0.flatten()

        # M-by-1 input loading
        # this is the direction that leads to most deflation in X
        p0 = np.matmul(C_XX, v0) / D
        p[:, i] = p0.flatten()

        # deflation.
        # let y' = y - q * z, x' = x - p * z
        # C_Y'X'
        # = E{y' * x'.T}
        # = (y - q*z) * (x - p*z).T
        # = y*x.T - q*z*x.T - y*z.T*p.T + q*z*z.T*p.T
        #
        # Plugging in z = v.T * x
        # = y*x.T - q*v.T*x*x.T - y*x.T*v*p.T + q*v.T*x*x.T*v*p.T
        # = C_YX - q*v.T*C_XX - C_YX*v*p.T + q*v.T*C_XX*v*P.T
        #
        # Noting that q * v.T * C_XX * v = C_YX * v
        # = C_YX - q*v.T*C_XX - q*v.T*C_XX*v*p.T + q*v.T*C_XX*v*p.T
        # = C_YX - q*v.T*C_XX
        # NOTE: this is different from the lecture notes
        #print(f"===== iteration {i} =====")
        #print(f"C_YX =\n{C_YX}")
        #print(f"C_XX =\n{C_XX}")
        #print(f"p0 =\n{p0}")
        #print(f"q0 =\n{q0}")
        #print(f"v0 =\n{v0}")

        X1 = X1 - np.matmul(p0, z0)
        Y1 = Y1 - np.matmul(q0, z0)
        """
        C_YX = C_YX - np.matmul(np.matmul(q0, v0.T), C_XX)
        C_XX = C_XX - np.matmul(np.matmul(p0, v0.T), C_XX)
        print("from iteration, C_YX =\n", C_YX)
        print("from deflation, C_YX =\n", np.matmul(Y1, X1.T) / (N - 1))
        """
        # FIXME: maybe it's better to just deflate samples and re-compute covariance
        # at each iteration
        C_YX = np.matmul(Y1, X1.T) / (N - 1)
        C_XX = np.matmul(X1, X1.T) / (N - 1)
        C_YY = np.matmul(Y1, Y1.T) / (N - 1)
        C_YY_history[i] = np.trace(C_YY) / C_YY_init_trace
        C_XX_history[i] = np.trace(C_XX) / C_XX_init_trace
        #print(f"iteration {i}")
        #print(f"  tr(C_YY) = {np.trace(C_YY)}")
        #print(f"  tr(C_XX) = {np.trace(C_XX)}")


    return (v, p, q, C_YY_history * 100, C_XX_history * 100)


def compress(X_raw, v, p, M_star):
    """
    :param
    :return X_compressed
    """
    assert(v.shape[0] == X_raw.shape[0])
    assert(v.shape[0] == p.shape[0])
    assert(v.shape[1] >= M_star)
    assert(p.shape[1] >= M_star)

    # make a copy
    X = copy.deepcopy(X_raw)

    # record the dimension
    (M, N) = X.shape

    # compress using the first M_star components
    X_compressed = np.zeros((M, N))
    for i in range(M_star):
        # M-by-1, the most dominant direction
        v0 = v[:, i].reshape(-1, 1)

        # M-by-1, corresponding input loading vector
        p0 = p[:, i].reshape(-1, 1)

        # 1-by-N, input score
        z0 = np.matmul(v0.T, X).reshape(1, -1)

        # M-by-N, contribution from this component
        x0 = np.matmul(p0, z0)

        # update the compressed value
        X_compressed += x0

        # deflate input
        X = X - np.matmul(p0, z0)

    return X_compressed


def predict(X_raw, v, p, q, M_star):
    """
    :return Y_predicted
    """
    assert(v.shape[0] == X_raw.shape[0])
    assert(v.shape[0] == p.shape[0])
    assert(v.shape[1] >= M_star)
    assert(p.shape[1] >= M_star)
    assert(q.shape[1] >= M_star)

    # make a copy
    X = copy.deepcopy(X_raw)

    # record the dimension
    (M, N) = X.shape
    (L, _) = q.shape

    # predict using the first M_star components
    Y_predicted = np.zeros((L, N))
    for i in range(M_star):
        # M-by-1, the most dominant direction
        v0 = v[:, i].reshape(-1, 1)

        # M-by-1, corresponding input loading vector
        p0 = p[:, i].reshape(-1, 1)

        # L-by-1, corresponding output loading vector
        q0 = q[:, i].reshape(-1, 1)

        # 1-by-N, input score
        z0 = np.matmul(v0.T, X).reshape(1, -1)

        # L-by-N, contribution from this component
        y0 = np.matmul(q0, z0)

        # update the prediction
        Y_predicted += y0

        # deflate input
        X = X - np.matmul(p0, z0)

    return Y_predicted


if (__name__ == "__main__"):
    import matplotlib.pyplot as plt

    print("Loading sample input and output")
    X = np.loadtxt("inputdata.txt", skiprows=1, dtype=float)
    Y = np.loadtxt("outputdata.txt", skiprows=1, dtype=float)

    # now each row is a sample input / output
    # we need to transpose X and Y so that each col is a sample input / output
    X = X.T
    Y = Y.T

    # NOTE: No normalization
    print("Running PLSR decomposition")
    (v, p, q, C_YY_history, C_XX_history) = decompose(X, Y, 3)

    M_star = 1
    print("Running PLSR compression")
    X_compressed = compress(X, v, p, M_star)
    X_error = X - X_compressed
    rmse = np.sqrt(np.mean(np.multiply(X_error, X_error), axis=1))
    print(f"Compression RMSE = {rmse}")

    print("Running PLSR prediction")
    Y_predicted = predict(X, v, p, q, M_star)
    Y_error = Y - Y_predicted
    rmse = np.sqrt(np.mean(np.multiply(Y_error, Y_error), axis=1))
    print(f"Prediction RMSE = {rmse}")

    plt.figure()
    for i in range(3):
        plt.subplot(3, 1, i + 1)
        plt.plot(X[i, :])
        plt.plot(X_compressed[i, :])
        plt.ylabel(f"X{i+1}")
        plt.legend(["raw", "compressed"])
    plt.suptitle(f"Raw and compressed input ({M_star} components)")

    plt.figure()
    for i in range(2):
        plt.subplot(3, 1, i + 1)
        plt.plot(Y[i, :])
        plt.plot(Y_predicted[i, :])
        plt.ylabel(f"Y{i+1}")
        plt.legend(["raw", "predicted"])
    plt.suptitle(f"Raw and predicted output ({M_star} components)")

    plt.figure()
    plt.plot(C_XX_history, "x-")
    plt.plot(C_YY_history, "o-")
    plt.legend(["input", "output"])
    plt.xlabel("component ID")
    plt.ylabel("residual var (%)")
    plt.title("Residual variance history")

    plt.show()
