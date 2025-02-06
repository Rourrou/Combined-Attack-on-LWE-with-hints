import numpy as np
from scipy.stats import binom, norm, uniform
import time
from proba import *
import math


def solve_perfect_hints_Del22(eta, a, b, max_nb_of_iterations=30, solution=None):
    [nb_of_hints, nb_of_unknowns] = a.shape
    guess = np.zeros(nb_of_unknowns, dtype=int)  # creat an initial guess of the solution with all values set to zero
    if nb_of_hints == 0:
        return guess
    nb_of_values = 2 * eta + 1
    x = np.arange(-eta, eta + 1, dtype=np.int8)
    x_pmf = binom.pmf(x + eta, 2 * eta, 0.5)
    # print("x_pmf: ", x_pmf)
    x_pmf = np.repeat(x_pmf.reshape(1, -1), nb_of_unknowns,
                      axis=0)  # this line repeats the x_pmf array multiple times to creat a 2D array

    a = a.astype(np.int16)  # this change the datatype of the matrix a to int16
    a_squared = np.square(a)  # this squares each element of a

    count = [0] * max_nb_of_iterations
    for z in range(max_nb_of_iterations):
        # print("Iteration " + str(z))
        time_start = time.time()
        mean = np.matmul(x_pmf, x)  # 计算当前分布下，所有未知数的期望值
        variance = np.matmul(x_pmf, np.square(x)) - np.square(mean)
        mean = np.multiply(a, np.repeat(mean[np.newaxis, :], nb_of_hints, axis=0))
        variance = np.multiply(a_squared, np.repeat(variance[np.newaxis, :], nb_of_hints, axis=0))
        mean = mean.sum(axis=1).reshape(-1, 1).repeat(nb_of_unknowns, axis=1) - mean  # 减去自身
        # print("mean",mean[:3])
        mean -= b[:, np.newaxis]
        # print("mean", mean[:3])
        variance = variance.sum(axis=1).reshape(-1, 1).repeat(nb_of_unknowns, axis=1) - variance
        variance = np.clip(variance, 1, None)
        psuccess = np.zeros((nb_of_values, nb_of_hints, nb_of_unknowns), dtype=float)
        for j in range(nb_of_values):
            # 求解连续高斯分布中某个点值的概率
            zscore = np.divide(a * x[j] + mean + 0.5, np.sqrt(variance))
            # zscore = np.divide(a * x[j] + mean, np.sqrt(variance))
            psuccess[j, :, :] = norm.cdf(zscore + 0.5) - norm.cdf(zscore - 0.5)  # Kyber128:0.5; Kyber256:1; Kyber512。

        psuccess = np.transpose(psuccess, axes=[2, 0, 1])
        psuccess = np.clip(psuccess, 10e-20, None)
        psuccess = np.sum(np.log(psuccess), axis=2)
        row_means = psuccess.max(axis=1)
        psuccess -= row_means[:, np.newaxis]
        psuccess = np.exp(psuccess)

        x_pmf = np.multiply(psuccess, x_pmf)
        row_sums = x_pmf.sum(axis=1)
        if np.count_nonzero(row_sums == 0) != 0:
            break
        x_pmf /= row_sums[:, np.newaxis]

        guess = x[np.argmax(x_pmf, axis=1)]

        # if z == max_nb_of_iterations - 1:
        #     print(np.array(guess))

        time_end = time.time()
        # print("Elapsed time: {:.1f} seconds".format(time_end - time_start))
        if solution is not None:
            nb_correct = np.count_nonzero(solution == guess)
            count[z] = nb_correct
            # print("Number of correctly guessed unknowns: {:d}/{:d}".format(nb_correct, len(solution)))
        if (z > 1) and count[z-1] >= count[z] + 2:
            # print(np.array(guess))
            count[z] = count[z - 1]
            break

    print("count", count)
    # print("guess", np.array(guess))
    short_vector = np.array(guess - solution)
    distance = np.linalg.norm(short_vector)
    distance = np.round(distance, 2)
    print("distance", distance)
    return guess, nb_correct, distance


def solve_ineq_hints_del22(eta, a, b, is_geq_zero, max_nb_of_iterations=20, solution=None):
    # print("Solving translated secret only perfect hints...")

    [nb_of_hints, nb_of_unknowns] = a.shape
    guess = np.zeros(nb_of_unknowns, dtype=int)  # creat an initial guess of the solution with all values set to zero

    if nb_of_hints == 0:
        print("the number of hints is 0 !")
        return guess
    nb_of_values = 2 * eta + 1
    x = np.arange(-eta, eta + 1, dtype=np.int8)
    x_pmf = binom.pmf(x + eta, 2 * eta, 0.5)
    # print("x_pmf", x_pmf)
    x_pmf = np.repeat(x_pmf.reshape(1, -1), nb_of_unknowns, axis=0)
    # x_pmf_static = x_pmf.copy()
    a = a.astype(np.int16)
    a_squared = np.square(a)

    count = [0] * max_nb_of_iterations
    for z in range(max_nb_of_iterations):
        #print("Iteration " + str(z))
        time_start = time.time()
        mean = np.matmul(x_pmf, x)  # 计算当前分布下，所有未知数的期望值
        variance = np.matmul(x_pmf, np.square(x)) - np.square(mean)
        mean = np.multiply(a, np.repeat(mean[np.newaxis, :], nb_of_hints, axis=0))
        variance = np.multiply(a_squared, np.repeat(variance[np.newaxis, :], nb_of_hints, axis=0))
        mean = mean.sum(axis=1).reshape(-1, 1).repeat(nb_of_unknowns, axis=1) - mean  # 减去自身
        mean -= b[:, np.newaxis]
        variance = variance.sum(axis=1).reshape(-1, 1).repeat(nb_of_unknowns, axis=1) - variance
        variance = np.clip(variance, 1, None)
        psuccess = np.zeros((nb_of_values, nb_of_hints, nb_of_unknowns), dtype=float)
        for j in range(nb_of_values):
            zscore = np.divide(a * x[j] + mean + 0.5, np.sqrt(variance))
            psuccess[j, :, :] = norm.cdf(zscore)  # central limit theorem

        psuccess = np.transpose(psuccess, axes=[2, 0, 1])
        psuccess = np.multiply(psuccess, is_geq_zero) + np.multiply(1 - psuccess, 1 - is_geq_zero)
        psuccess = np.clip(psuccess, 10e-5, None)
        psuccess = np.sum(np.log(psuccess), axis=2)
        row_means = psuccess.max(axis=1)
        psuccess -= row_means[:, np.newaxis]
        psuccess = np.exp(psuccess)

        # x_pmf = np.multiply(psuccess, x_pmf_static) # SMY:20241009
        x_pmf = np.multiply(psuccess, x_pmf)  # SMY:20241009
        # print("x_pmf_before_nor", x_pmf[0, :])
        row_sums = x_pmf.sum(axis=1)
        x_pmf /= row_sums[:, np.newaxis]
        guess = x[np.argmax(x_pmf, axis=1)]
        # print("guess", np.array(guess))

        # if z == max_nb_of_iterations - 1:
        #     print(np.array(guess))

        time_end = time.time()
        #print("Elapsed time: {:.1f} seconds".format(time_end - time_start))

        if solution is not None:
            nb_correct = np.count_nonzero(solution == guess)
            count[z] = nb_correct
            # print("Number of correctly guessed unknowns: {:d}/{:d}".format(nb_correct, len(solution)))

        if (z > 1) and count[z - 1] >= count[z] + 2:
            # print(np.array(guess))
            count[z] = count[z - 1]
            break
    print("count", count)
    # print("guess", np.array(guess))
    # short_vector = np.concatenate((np.array(guess - solution), np.array([1])))
    short_vector = np.array(guess - solution)
    distance = np.linalg.norm(short_vector)
    distance = np.round(distance, 2)
    print("distance", distance)
    return guess, nb_correct, distance


def solve_approx_hints_Del22(eta, sigma, V, L, max_nb_of_iterations=20, solution=None, so_flag=None):
    # if so_flag:
    #     print("Solving secret only approximate hints...")
    # else:
    #     print("Solving secret error approximate hints...")

    [nb_of_hints, nb_of_unknowns] = V.shape
    guess = np.zeros(nb_of_unknowns, dtype=int)  # creat an initial guess of the solution with all values set to zero
    if nb_of_hints == 0:
        return guess
    nb_of_values = 2 * eta + 1
    x = np.arange(-eta, eta + 1, dtype=np.int8)
    x_pmf = binom.pmf(x + eta, 2 * eta, 0.5)
    # print("x_pmf: ", x_pmf)
    x_pmf = np.repeat(x_pmf.reshape(1, -1), nb_of_unknowns, axis=0)

    V = V.astype(np.int16)
    V_squared = np.square(V)

    count = [0] * max_nb_of_iterations
    for z in range(max_nb_of_iterations):
        # print("Iteration " + str(z))
        time_start = time.time()
        mean = np.matmul(x_pmf, x)  # 计算当前分布下，所有未知数的期望值
        variance = np.matmul(x_pmf, np.square(x)) - np.square(mean)
        mean = np.multiply(V, np.repeat(mean[np.newaxis, :], nb_of_hints, axis=0))
        variance = np.multiply(V_squared, np.repeat(variance[np.newaxis, :], nb_of_hints, axis=0))
        mean = mean.sum(axis=1).reshape(-1, 1).repeat(nb_of_unknowns, axis=1) - mean  # 减去自身
        mean -= L[:, np.newaxis]
        variance = variance.sum(axis=1).reshape(-1, 1).repeat(nb_of_unknowns, axis=1) - variance
        variance = np.clip(variance, 1, None)
        psuccess = np.zeros((nb_of_values, nb_of_hints, nb_of_unknowns), dtype=float)
        for j in range(nb_of_values):
            # 求解连续高斯分布中某个点值的概率
            zscore_pos = np.divide(V * x[j] + mean + 3*sigma, np.sqrt(variance))
            zscore_neg = np.divide(V * x[j] + mean - 3*sigma, np.sqrt(variance))
            psuccess[j, :, :] = norm.cdf(zscore_pos) - norm.cdf(zscore_neg)  # central limit theorem

        psuccess = np.transpose(psuccess, axes=[2, 0, 1])
        # print("psuccess", psuccess[0])
        psuccess = np.clip(psuccess, 10e-10, None)
        psuccess = np.sum(np.log(psuccess), axis=2)
        # print("psuccess", psuccess)
        row_means = psuccess.max(axis=1)
        psuccess -= row_means[:, np.newaxis]
        # print("row_means",row_means)
        psuccess = np.exp(psuccess)

        x_pmf = np.multiply(psuccess, x_pmf)
        row_sums = x_pmf.sum(axis=1)
        x_pmf /= row_sums[:, np.newaxis]
        guess = x[np.argmax(x_pmf, axis=1)]

        # if z == max_nb_of_iterations - 1:
        #     print(np.array(guess))

        time_end = time.time()
        # print("Elapsed time: {:.1f} seconds".format(time_end - time_start))
        if solution is not None:
            nb_correct = np.count_nonzero(solution == guess)
            count[z] = nb_correct
            # print("Number of correctly guessed unknowns: {:d}/{:d}".format(nb_correct, len(solution)))
        if (z > 1) and count[z-1] >= count[z] + 1:
            # print(np.array(guess))
            count[z] = count[z - 1]

            break

    print("count", count)
    # print("guess", np.array(guess))
    # short_vector = np.concatenate((np.array(guess - solution), np.array([1])))
    short_vector = np.array(guess - solution)
    distance = np.linalg.norm(short_vector)
    distance = np.round(distance, 2)
    print("distance", distance)
    return guess, nb_correct, distance
