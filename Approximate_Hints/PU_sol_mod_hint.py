#!/usr/bin/python3
import random
import numpy as np
import time
from tqdm import tqdm
from scipy.stats import binom, norm, uniform


def sol_perfect_hints_with_prob(eta, V, L, max_nb_of_iterations=15):
    print("Solving secret only perfect hints with probability...")

    [nb_of_hints, nb_of_unknowns] = V.shape
    guess = np.zeros((nb_of_unknowns), dtype=int)  # creat an initial guess of the solution with all values set to zero
    if nb_of_hints == 0:
        return guess
    nb_of_values = 2 * eta + 1
    x = np.arange(-eta, eta + 1, dtype=np.int8)
    x_pmf = binom.pmf(x + eta, 2 * eta, 0.5)
    # print("x_pmf: ", x_pmf)
    x_pmf = np.repeat(x_pmf.reshape(1, -1), nb_of_unknowns, axis=0)
    V = V.astype(np.int32)
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
            # zscore_pos = np.divide(a * x[j] + mean + 0.5 + 100, np.sqrt(variance))
            # zscore_neg = np.divide(a * x[j] + mean + 0.5 - 100, np.sqrt(variance))
            # psuccess[j, :, :] = norm.cdf(zscore_pos) - norm.cdf(zscore_neg)  # central limit theorem
            # SMY：连续分布无法计算d=zscore的概率，使用cdf(zscore+1)-cdf(zscore-1)
            zscore = np.divide(-a * x[j] - mean, np.sqrt(variance))
            # print("zscore", zscore)
            psuccess[j, :, :] = norm.cdf(zscore + 0.5) - norm.cdf(
                zscore - 0.5)  # Kyber128:0.5; Kyber256:1; Kyber512。
        # print("psuccess[, 0, 0]",psuccess[:, 0, 0])

        psuccess = np.transpose(psuccess, axes=[2, 0, 1])
        # print("psuccess", psuccess[0])
        psuccess = np.clip(psuccess, 10e-20, None)
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

        if z == max_nb_of_iterations - 1:
            print(np.array(guess))

        time_end = time.time()
        # print("Elapsed time: {:.1f} seconds".format(time_end - time_start))
        if solution is not None:
            nb_correct = np.count_nonzero(solution == guess)
            count[z] = nb_correct
            # print("Number of correctly guessed unknowns: {:d}/{:d}".format(nb_correct, len(solution)))
        if (z > 1) and count[z - 1] >= count[z] + 1:
            # print(np.array(guess))
            count[z] = count[z - 1]

            break
    print("count", count)
    print("guess", np.array(guess))
    short_vector = np.concatenate((np.array(guess - solution), np.array([1])))
    distance = np.linalg.norm(short_vector)
    distance = np.round(distance, 2)
    print("distance", distance)
    return guess, count[z], distance

def mod2perf(V, L, eta, q):
    [nb_of_hints, nb_of_unknowns] = V.shape

    nb_of_values = 2 * eta + 1
    x = np.arange(-eta, eta + 1, dtype=np.int8)
    x_pmf = binom.pmf(x + eta, 2 * eta, 0.5)
    print("x_pmf: ", x_pmf)
    x_pmf = np.repeat(x_pmf.reshape(1, -1), nb_of_unknowns,
                      axis=0)  # this line repeats the x_pmf array multiple times to creat a 2D array

    V = V.astype(np.int32)  # this change the datatype of the matrix a to int16
    V_squared = np.square(V)  # this squares each element of a

    mean = np.matmul(x_pmf, x)  # 计算当前分布下，所有未知数的期望值
    variance = np.matmul(x_pmf, np.square(x)) - np.square(mean)
    mean = np.multiply(V, np.repeat(mean[np.newaxis, :], nb_of_hints, axis=0))
    variance = np.multiply(V_squared, np.repeat(variance[np.newaxis, :], nb_of_hints, axis=0))
    mean = mean.sum(axis=1).reshape(-1, 1)
    variance = variance.sum(axis=1).reshape(-1, 1)
    variance = np.clip(variance, 1, None)
    # print("variance",variance)

    low_prob = 10e-2
    V_ext = []
    L_ext = []

    for i in range(nb_of_hints):
        start = 0
        prob_l = prob_r = 1
        while prob_l >= low_prob and prob_r >= low_prob:
            if start == 0:
                lower = (-(q / 2) - mean[i]) / np.sqrt(variance[i])
                upper = ( q / 2 - mean[i]) / np.sqrt(variance[i])
                prob = norm.cdf(upper) - norm.cdf(lower)
                V_ext.append(V[i])
                L_ext.append([(L[i], prob)])
            else:
                lower_l = (-(q*start + q / 2) - mean[i]) / np.sqrt(variance[i])
                upper_l = (-(q*start - q / 2) - mean[i]) / np.sqrt(variance[i])
                prob_l = norm.cdf(upper_l) - norm.cdf(lower_l)
                V_ext.append(V[i])
                L_ext.append([(-q*start +L[i], prob_l)])

                lower_r = (q*start - q / 2 - mean[i]) / np.sqrt(variance[i])
                upper_r = (q*start + q / 2 - mean[i]) / np.sqrt(variance[i])
                prob_r = norm.cdf(upper_r) - norm.cdf(lower_r)
                V_ext.append(V[i])
                L_ext.append([(q * start + L[i], prob_r)])
            start += 1

    return V_ext, L_ext


def sol_mod_q_hints(m, k, solution):
    ETA = 3
    q = 3329

    nb_of_hints = 1280
    nb_of_unknowns = len(solution)
    print("The number of unknowns", nb_of_unknowns)

    with open("Data/Mod_q_hints/secret-only/Kyber128/v.txt", 'r') as f:
        lines_V = [next(f) for _ in range(nb_of_hints)]
    V = np.loadtxt(lines_V)

    with open("Data/Mod_q_hints/secret-only/Kyber128/l.txt", 'r') as g:
        lines_L = [next(g) for _ in range(nb_of_hints)]
    L = np.loadtxt(lines_L)
    # print(b)

    if m == 0:
        guess = [0] * nb_of_unknowns
        nb_correct = np.count_nonzero(solution == guess)
        print("\nThe average recovered coefficients with %d hints is %d" % (m, nb_correct))
        short_vector = np.concatenate((np.array(guess - solution), np.array([1])))
        distance = np.linalg.norm(short_vector)
        distance = np.round(distance, 2)
        return nb_correct, distance, 0

    rec_num = []
    rec_dis = []
    num_correct = 0 # k次实验中，正确恢复完整私钥的次数
    # k次实验
    for i in range(k):
        # 选择m个索引
        indices = random.sample(range(nb_of_hints), m)
        V_selected = np.array([V[j] for j in indices])
        L_selected = np.array([L[j] for j in indices])

        V_ext, L_ext = mod2perf(V_selected, L_selected, ETA, q)
        print("L_ext", L_ext)
    #     print("the num of extend hints is ", len(V_ext))
    #
    #     s, n, d = solve_perfect_hints_with_prob(ETA, a_selected, b_selected, solution=solution, so_flag=None)
    #
    #     rec_num.append(n)
    #     rec_dis.append(d)
    #     if n == nb_of_unknowns:
    #         num_correct += 1
    #     print("Number of selected coeffs matches:{:d}/{:d}".format(n, nb_of_unknowns))
    #
    # ave_rec_num = np.mean(rec_num)
    # ave_rec_dis = np.round(np.mean(rec_dis), 2)
    # success_rate = num_correct / k
    #
    # print("the average recovered coefficients with %d ineqs is %f/%d" % (m, ave_rec_num, nb_of_unknowns))
    # print("the success prob of recovering full ineqs with %d ineqs is %f" % (m, success_rate))

    return ave_rec_num, ave_rec_dis, success_rate



if __name__ == "__main__":
    with open("Data/Mod_q_hints/secret-only/Kyber128/s.txt", 'r') as g:
        solution = g.readlines()
    solution = np.array([int(x) for x in solution[0].split()])
    print("solution", solution)

    num_ine = []
    num_rec = []
    dis_rec = []
    suc_rat = []

    for m in tqdm(range(50,51,50)):
        num_ine.append(m)
        print("\nThe number of mod_q hint is", m)
        rec, dis, ratio = sol_mod_q_hints(m, 30, solution)
        num_rec.append(rec)
        dis_rec.append(dis)
        suc_rat.append(ratio)

    print("num_ine: ", num_ine)
    print("num_rec: ", [round(x,1) for x in num_rec])
    print("dis_rec: ", dis_rec)
    print("suc_rat: ", suc_rat)


