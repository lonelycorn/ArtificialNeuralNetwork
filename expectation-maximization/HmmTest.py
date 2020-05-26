import numpy as np

"""
Discrete hidden states x[k] with
* Time-independent stochastic state transition matrix
* Time-independent stochastic observation matrix
* Initial state distribution

Goal: given a sequence of measurements, estimate most likely
* transition matrix
* observation matrix
* initial state distribution
* sequence of state

Note: it's not uncommon to find the recovered sequence different from the ground
truth, which is due to overfitting

ref: https://www.ece.ucsb.edu/Faculty/Rabiner/ece259/Reprints/tutorial%20on%20hmm%20and%20applications.pdf
"""

def drawSample(cumulativeProbability):
    M = len(cumulativeProbability)
    p = np.random.sample() # between [0, 1)
    for i in range(M):
        if (cumulativeProbability[i] >= p):
            return i
    return M - 1


def generateGroundTruthSequence(N, transitionProbability, firstStateProbability):
    """
    :param N: number of samples to generate
    :param transitionProbability:   M-by-M matrix. each row gives probability to transit to other states
    :return groundTruthSequence
    """
    M = len(transitionProbability)
    cumulativeProbability = np.cumsum(transitionProbability, axis=1)
    #print("cumulative transition probability")
    #print(cumulativeProbability)
    groundTruthSequence = np.zeros(N, dtype=np.int)

    k = drawSample(np.cumsum(firstStateProbability))
    groundTruthSequence[0] = k
    for i in range(1, N):
        k = drawSample(cumulativeProbability[k, :])
        groundTruthSequence[i] = k

    return groundTruthSequence


def generateMeasurements(groundTruthSequence, observationProbability):
    """
    :param observationProbability: M-by-M matrix
    :return measurements
    """
    cumulativeProbability = np.cumsum(observationProbability, axis=1)
    #print("cumulative observation probability")
    #print(cumulativeProbability)
    measurements = [drawSample(cumulativeProbability[k, :]) for k in groundTruthSequence]
    return np.array(measurements)


def computeTransitionLikelihood(sequence, M):
    N = len(sequence)
    count = np.zeros((M, M), dtype=np.int)
    for i in range(N - 1):
        t = sequence[i]
        s = sequence[i + 1]
        count[t, s] += 1

    print("transition count")
    print(count)

    likelihood = np.zeros(count.shape, np.float)
    for i in range(M):
        d = np.sum(count[i, :])
        # to avoid division-by-0
        if (d > 0):
            likelihood[i, :] = count[i, :] / d
    return likelihood


def computeObservationLikelihood(sequence, measurements, M):
    assert(len(sequence) == len(measurements))

    N = len(sequence)
    count = np.zeros((M, M), dtype=np.int)
    for (s, z) in zip(sequence, measurements):
        count[s, z] += 1

    print("observation count")
    print(count)

    likelihood = np.zeros(count.shape, np.float)
    for i in range(M):
        d = np.sum(count[i, :])
        # to avoid division-by-0
        if (d > 0):
            likelihood[i, :] = count[i, :] / d
    return likelihood


def findMostLikelySequence(transitionProbability, observationProbability, firstStateProbability, measurements):
    """
    Vertibi algorithm
    :return (mostLikelySequence, logLikelihood)
    """
    assert(transitionProbability.shape == observationProbability.shape)
    assert(len(transitionProbability) == len(firstStateProbability))

    N = len(measurements)
    M = len(transitionProbability)

    # When T and O are fixed & given
    # P(z[0] ... z[k-1], z[k], x[0] ... x[k-1], x[k])
    # = P(z[k] | z[0] ... z[k-1], x[0] ... x[k-1], x[k]) *
    #   P(x[k] | z[0] ... z[k-1], x[0] ... x[k-1]) *
    #   P(z[0] ... z[k-1], x[0] ... x[k-1])
    # = P(z[k] | x[k]) * P(x[k] | x[k-1]) * P(z[0] ... z[k-1], x[0] ... x[k-1])
    #
    # The last step originates from the Markov property, which suggests thatz[k]
    # is only dependent of x[k], and x[k] is determined solely by x[k-1]
    #
    # Let
    #   F[k, s] = log(P(z[0] ... z[k-1], z[k], x[0] ... x[k-1], x[k]=s))
    #   T[k, t, s] = log(P(x[k]=s | x[k-1]=t))
    #   O[k, s, r] = log(P(z[k]=r | x[k]=s))
    # Note we could drop index k from T and O because these two matrices do NOT
    # change over time
    #
    # Then we have
    #   F[0, s] = log(firstStateProbability[s]) + O[s, z[0]]
    #   F[k, s] = max{ O[s, z[k]] + T[t, s] + F[k-1, t] }

    T = np.log(transitionProbability)
    O = np.log(observationProbability)
    F = np.zeros((N, M), dtype=np.float)
    prev = np.zeros(F.shape, dtype=np.int) # assignment of x[k-1] that maximizes F[k, s]
    z = measurements

    for s in range(M):
        F[0, s] = O[s, z[0]] + np.log(firstStateProbability[s])
        prev[0, s] = -1
    #print(f"k = 0, F[k, :] = {np.exp(F[0, :])}")

    for k in range(1, N):
        for s in range(M): # current state
            best = -np.inf
            idx = -1
            for t in range(M): # previous state
                proposal = O[s, z[k]] + T[t, s] + F[k - 1, t]
                if (proposal > best):
                    best = proposal
                    idx = t
                #print(f"k = {k}, t = {t}, s = {s}, z = {z[k]}, value = {np.exp(proposal)}")
            # NOTE: sometimes idx < 0 because state s at step k is infeasible; when
            # this happens, we'll leave F[k, s] = -inf, and prev[k, s] = -1
            F[k, s] = best
            prev[k, s] = idx
        #print(f"k = {k}, F[k, :] = {np.exp(F[k, :])}")

    #print(f"F\n{F}")
    #print(f"prev\n{prev}")

    # recover most likely sequence
    mostLikelySequence = np.zeros(N, dtype=np.int)
    s = np.argmax(F[N - 1, :])
    mostLikelySequence[N - 1] = s
    logLikelihood = F[N-1, s]
    for k in range(N - 1, 0, -1):
        s = prev[k, s]
        mostLikelySequence[k - 1] = s

    return (mostLikelySequence, logLikelihood)


def findMostLikelyParameters(measurements,
        M,
        transitionLikelihood=None,
        observationLikelihood=None,
        firstStateLikelihood=None):
    """
    Baum-Welsh (EM) algorithm
    Estimate the most likely parameters given the measurements using Baum-Welsh (EM) algorithm
    :return transitionLikelihood, observationLikelihood, firstStateLikelihood
    """
    EPSILON = 1e-4
    INITIAL_OBSERVATION_ERROR_RATE = 0.3

    def pertubateLikelihood(likelihood, epsilon=EPSILON):
        M = len(likelihood)
        return likelihood * (1.0 - epsilon) + np.ones(likelihood.shape) * epsilon / M

    N = len(measurements)

    # initialization
    if (transitionLikelihood is None):
        transitionLikelihood = computeTransitionLikelihood(measurements, M)
    if (observationLikelihood is None):
        observationLikelihood = pertubateLikelihood(np.eye(M), epsilon=INITIAL_OBSERVATION_ERROR_RATE)
    if (firstStateLikelihood is None):
        firstStateLikelihood = np.array([np.sum(measurements == i) for i in range(M)]) / N

    print(f"initial transition likelihood\n{transitionLikelihood}")
    print(f"initial observation likelihood\n{observationLikelihood}")
    print(f"initial likelihood\n{firstStateLikelihood}")


    count = 0
    prevLogLikelihood = 0
    while (True):
        count += 1

        T = transitionLikelihood
        O = observationLikelihood
        pi = firstStateLikelihood
        z = measurements

        # Let X = {x[0] ... x[N-1]} denote the trajectory of the hidden states, and
        # Z = {z[0] ... z[N-1]} all available measurements
        #
        # Let lambda = {T; O; pi} represent all model parameters, where
        #   T[i, s, t] = Pr(x[i+1] == t | x[i] == s)
        #   O[i, s, t] = Pr(z[i] == t | x[i] == s)
        #   pi[s] = Pr(x[0] == s)
        # Note that index i could be droped from O and T because both state transition
        # probability and observation probability are time-invariant
        #
        # The joint distribution of hidden state at step i, and all measurements
        #   P(x[i], z[0] ... z[N-1] | lamda)
        # = P(z[i+1] ... z[N-1] | x[i], z[0] ... z[i], lambda) * P(x[i], z[0] ... z[i] | lambda)
        #
        # From the HMM factor graph, it could be seen that future observations are
        # conditionally independent of past observations, given the current hidden
        # state, i.e.
        #   P(z[i+1] ... z[N-1] | x[i], z[0] ... z[i], lambda) = P(z[i+1] ... z[N-1] | x[i], lambda)
        #
        # With the above line, the original joint distribution could be simplified to
        #   P(x[i], z[0] ... z[N-1] | lambda)
        # = P(z[i+1] ... z[N-1] | x[i], lambda) * P(x[i], z[0] ... z[i] | labmda)
        #
        # Define
        #   alpha[i, s] = Pr(x[i]=s, z[0] ... z[i] | lambda)
        #   beta[i, s] = Pr(z[i+1] ... z[N-1] | x[i]=s, lambda)
        #
        # We could compute alpha and beta by induction as follows
        #   alpha[0, s] = pi[s] * O[s, z[0]]
        #   alpha[i, s] = sum(alpha[i-1, t] * T[t, s]) * O[s, z[i]]
        #   beta[N-1, s] = 1
        #   beta[i, s] = sum(beta[i+1, t] * T[s, t] * O[t, z[i+1]]
        #
        # The probability of being in a specific state at step i, given all measurements and
        # all model parameters could then be written using alpha and beta as
        #   gamma[i, s]  = Pr(x[i]=s | z[0] ... z[N-1], lambda)
        # = Pr(x[i]=s, z[0] ... z[N-1], lambda) / Pr(z[0] ... z[N-1] | lambda)
        # = Pr(z[i+1] ... z[N-1] | x[i]=s, lambda) * Pr(x[i]=s, z[0] ... z[i] | lambda) / Pr(z[0] ... z[N-1] | lambda)
        # = alpha[i, s] * beta[i, s] / sum(alpha[i, t] * beta[i, t])
        #
        # The probability of transiting from one state to another at step i, given all
        # measurements and all model parameters is
        #   ksi[i, s, t] = Pr(x[i]=s, x[i+1]=t | z[0] ... z[N-1], labmda)
        # = Pr(x[i]=s, x[i+1]=t, z[0] ... z[N-1] | lambda) / P(z[0] ... z[N-1], lambda)
        # = Pr(z[i+2] ... z[N-1] | x[i]=s, x[i+1]=t, z[0] ... z[i+1], lambda) *
        #   Pr(z[i+1] | x[i]=s], x[i+1]=t, z[0] ... z[i], lambda) *
        #   Pr(x[i+1]=t | x[i]=s, z[0] ... z[i], lambda) *
        #   Pr(x[i]=s,  z[0] ... z[i] | lambda) /
        #   Pr(z[0] ... z[N-1] | lambda)
        # = Pr(z[i+2] ... z[N-1] | x[i+1]=t, lambda) *
        #   Pr(z[i+1] | x[i+1]=t) *
        #   Pr(s[i+1]=t | x[i]=s) *
        #   Pr(x[i]=s, z[0] ... z[i] | lambda) /
        #   Pr(z[0] ... z[N-1] | lambda)
        # = beta[i+1, t] * O[t, z[i+1]] * T[s, t] * alpha[i, s] / sum(alpha[i, k] * beta[i, k])

        # forward pass, with scaling
        alpha = np.zeros((N, M))
        c = np.zeros(N) # scaling factor at each step
        for s in range(M):
            alpha[0, s] = pi[s] * O[s, z[0]]
        c[0] = 1.0 / np.sum(alpha[0, :])
        alpha[0, :] *= c[0]

        for i in range(1, N):
            for s in range(M):
                for t in range(M):
                    alpha[i, s] += alpha[i-1, t] * T[t, s] * O[s, z[i]]
            c[i] = 1.0 / np.sum(alpha[i, :])
            alpha[i, :] *= c[i]


        # backward pass, with scaling
        beta = np.zeros((N, M))
        beta[N-1, :] = 1.0
        for i in range(N-2, -1, -1):
            for s in range(M):
                for t in range(M):
                    beta[i, s] += beta[i+1, t] * T[s, t] * O[t, z[i+1]]

            beta[i, :] *= c[i]

        # NOTE: due to scaling of alpha and beta, it no longer holds that
        #   P(z[0] ... z[k] | lambda) = sum(alpha[i, s] * beta[i, s]) for any i
        # as such, we have to perform normalization at each step to turn the
        # corresponding terms a probability measure

        # compute gamma, i.e. P(x[i] | z[0] ... z[N-1], lambda)
        # NOTE: due to scaling the denominators have to be computed at each step
        gamma = alpha * beta
        gamma = (gamma.T / np.sum(gamma, axis=1)).T

        # compute ksi, i.e. P(x[i], x[i+1] | z[0] ... z[N-1], lambda)
        # NOTE: due to scaling the denominators have to be computed at each step
        ksi = np.zeros((N-1, M, M))
        for i in range(N-1):
            for s in range(M):
                for t in range(M):
                    ksi[i, s, t] = alpha[i, s] * T[s, t] * O[t, z[i+1]] * beta[i+1, t]
            ksi[i, :, :] /= np.sum(ksi[i, :, :])

        # update initial distribution
        firstStateLikelihood = gamma[0, :]

        # update transition probability
        for s in range(M): # current state
            for t in range(M): # next state
                denom = np.sum(gamma[:N-1, s])
                if (denom > 0):
                    transitionLikelihood[s, t] = np.sum(ksi[:, s, t]) / denom

        # update observation probability
        for t in range(M): # observation
            indices = (z == t)
            for s in range(M): # current state
                denom = np.sum(gamma[:N, s])
                if (denom > 0):
                    observationLikelihood[s, t] = np.sum(gamma[indices, s]) / denom


        # assuming true probability == current likelihood
        # NOTE: add some small pertubation to avoid taking logarithm of 0
        (mostLikelySequence, logLikelihood) = findMostLikelySequence(
                pertubateLikelihood(transitionLikelihood),
                pertubateLikelihood(observationLikelihood),
                pertubateLikelihood(firstStateLikelihood),
                measurements)
        #print(f"\n\niteration {count}")
        #print(f"\nc\n{c}")
        #print(f"\nalpha\n{alpha}")
        #print(f"\nbeta\n{beta}")
        #print(f"\ngamma\n{gamma}")
        #print(f"\nksi\n{ksi}")
        #print(f"\tcurrent transition likelihood\n{transitionLikelihood}")
        #print(f"\tcurrent observation likelihood\n{observationLikelihood}")
        #print(f"\tcurrent initial likelihood\n{firstStateLikelihood}")
        #print(f"\tcurrent sequence\n{mostLikelySequence}")
        #print(f"\tcurrent sequence log likelihood = {logLikelihood}")


        if (np.abs(prevLogLikelihood - logLikelihood) < EPSILON):
            print(f"Converged with {count} iterations")
            break
        prevLogLikelihood = logLikelihood

    return (transitionLikelihood, observationLikelihood, firstStateLikelihood)


if (__name__ == "__main__"):
    # serious example
    N = 100
    M = 3
    transitionProbability = np.array([[0.1, 0.5, 0.4], [0.2, 0.1, 0.7], [0.6, 0.3, 0.1]])
    observationProbability = np.array([[0.85, 0.1, 0.05], [0.05, 0.90, 0.05], [0.1, 0.1, 0.8]])
    firstStateProbability = np.array([0.2, 0.5, 0.4])

    groundTruthSequence = generateGroundTruthSequence(N, transitionProbability, firstStateProbability)
    print(f"ground truth sequence\n{groundTruthSequence}")

    measurements = generateMeasurements(groundTruthSequence, observationProbability)
    print(f"measurements\n{measurements}")
    difference = np.sum(measurements != groundTruthSequence)
    print(f"measurements error rate = {difference / N * 100}% ({difference} / {N})")

    (transitionLikelihood, observationLikelihood, firstStateLikelihood) = findMostLikelyParameters(measurements, M)

    print(f"estimated transition probability\n{transitionLikelihood}")
    print(f"ground truth transition probability\n{transitionProbability}")
    print(f"estimated observation probability\n{observationLikelihood}")
    print(f"ground truth observation probability\n{observationProbability}")
    print(f"estimated first state probability\n{firstStateLikelihood}")
    print(f"ground truth first state probability\n{firstStateProbability}")

    (sequence, _) = findMostLikelySequence(transitionLikelihood, observationLikelihood, firstStateProbability, measurements)

    difference = np.sum(sequence != groundTruthSequence)
    print(f"recovered sequence\n{sequence}")
    print(f"sequence error rate = {difference / N * 100}% ({difference} / {N})")


    """
    # EM test
    # example taken from https://ocw.mit.edu/courses/aeronautics-and-astronautics/16-410-principles-of-autonomy-and-decision-making-fall-2010/lecture-notes/MIT16_410F10_lec21.pdf
    N = 20
    M = 3 # 0: LA; 1: NY; 2: null
    measurements = np.array([2, 0, 0, 2, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 2, 0, 0, 1])
    transitionLikelihood = np.array([[0.5, 0.5, 0], [0.5, 0.5, 0], [0, 0, 0]])
    observationLikelihood = np.array([[0.4, 0.1, 0.5], [0.1, 0.5, 0.4], [0, 0, 1]])
    firstStateLikelihood = np.array([0.5, 0.5, 0.0])
    (transitionLikelihood, observationLikelihood, firstStateLikelihood) = findMostLikelyParameters(
            measurements,
            M,
            transitionLikelihood,
            observationLikelihood,
            firstStateLikelihood)
    print(f"estimated transition probability\n{transitionLikelihood}")
    print(f"estimated observation probability\n{observationLikelihood}")
    print(f"estimated first state probability\n{firstStateLikelihood}")
    """


    """
    # Viterbi test
    # example taken from https://ocw.mit.edu/courses/aeronautics-and-astronautics/16-410-principles-of-autonomy-and-decision-making-fall-2010/lecture-notes/MIT16_410F10_lec21.pdf
    transitionProbability = np.array([[0, 0.5, 0.5], [0, 0.9, 0.1], [0, 0, 1]])
    observationProbability = np.array([[0, 0.5, 0.5], [0, 0.9, 0.1], [0, 0.1, 0.9]])
    firstStateProbability = np.array([1, 0, 0])
    measurements = np.array([1, 2, 2, 1, 1, 1, 2, 1, 2])
    (sequence, _) = findMostLikelySequence(
            transitionProbability,
            observationProbability,
            firstStateProbability,
            measurements)

    print(f"most likely sequence: {sequence}"))
    """

