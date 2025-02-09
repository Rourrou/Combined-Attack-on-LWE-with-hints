import random
import numpy as np
import time
from tqdm import tqdm
from scipy.stats import binom, norm, uniform


def sol_perfect_hints_with_prob(eta, V, L, P, solution, max_nb_of_iterations=15):
    print("Solving secret only perfect hints with probability...")

    V = np.array(V)
    [nb_of_hints, nb_of_unknowns] = V.shape
    guess = np.zeros(nb_of_unknowns, dtype=int)  # creat an initial guess of the solution with all values set to zero
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
        print("Iteration " + str(z))
        time_start = time.time()
        mean = np.matmul(x_pmf, x)  # 计算当前分布下，所有未知数的期望值
        variance = np.matmul(x_pmf, np.square(x)) - np.square(mean)
        mean = np.multiply(V, np.repeat(mean[np.newaxis, :], nb_of_hints, axis=0))
        # print("mean", mean)
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
            zscore = np.divide(-V * x[j] - mean, np.sqrt(variance))
            # print("zscore", zscore)
            psuccess[j, :, :] = (norm.cdf(zscore + 0.5) - norm.cdf(zscore - 0.5)) * P[j]  # Kyber128:0.5; Kyber256:1; Kyber512。
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
    #print("variance",variance)

    low_prob = 50e-2
    V_ext = []
    L_ext = []
    P_ext = []

    for i in range(nb_of_hints):
        start = 0
        prob_l = prob_r = 1
        while prob_l >= low_prob and prob_r >= low_prob:
            if start == 0:
                lower = (-(q / 2) - mean[i]) / np.sqrt(variance[i])
                upper = ( q / 2 - mean[i]) / np.sqrt(variance[i])
                prob = norm.cdf(upper) - norm.cdf(lower)
                V_ext.append(V[i])
                L_ext.append(L[i])
                P_ext.append(prob)

            else:
                lower_l = (-(q*start + q / 2) - mean[i]) / np.sqrt(variance[i])
                upper_l = (-(q*start - q / 2) - mean[i]) / np.sqrt(variance[i])
                prob_l = norm.cdf(upper_l) - norm.cdf(lower_l)
                if prob_l >= low_prob:
                    V_ext.append(V[i])
                    L_ext.append(-q * start + L[i])
                    P_ext.append(prob_l)

                lower_r = (q*start - q / 2 - mean[i]) / np.sqrt(variance[i])
                upper_r = (q*start + q / 2 - mean[i]) / np.sqrt(variance[i])
                prob_r = norm.cdf(upper_r) - norm.cdf(lower_r)
                if prob_r >= low_prob:
                    V_ext.append(V[i])
                    L_ext.append(q * start + L[i])
                    P_ext.append(prob_r)
            start += 1

    return V_ext, L_ext, P_ext


def sol_mod_q_hints(m, q, k, solution):
    ETA = 3
    nb_of_hints = 2000
    nb_of_unknowns = len(solution)
    print("The number of unknowns", nb_of_unknowns)

    with open("Data/Modular Hints/Mod_k/secret error/Kyber128/k50/v.txt", 'r') as f:
        lines_V = [next(f) for _ in range(nb_of_hints)]
    V = np.loadtxt(lines_V)

    with open("Data/Modular Hints/Mod_k/secret error/Kyber128/k50/l.txt", 'r') as g:
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

        V_ext, L_ext, P_ext = mod2perf(V_selected, L_selected, ETA, q)
        L_ext = np.array(L_ext)
        P_ext = np.array(P_ext)

        # print("V_ext", V_ext)
        # print("L_ext", L_ext)
        # print("P_ext", P_ext)
        print("the num of extend hints is ", len(V_ext))

        s, n, d = sol_perfect_hints_with_prob(ETA, V_ext, L_ext, P_ext, solution)
        rec_num.append(n)
        rec_dis.append(d)
        if n == nb_of_unknowns:
            num_correct += 1
        print("Number of selected coeffs matches:{:d}/{:d}".format(n, nb_of_unknowns))

    ave_rec_num = np.mean(rec_num)
    ave_rec_dis = np.round(np.mean(rec_dis), 2)
    success_rate = num_correct / k

    print("the average recovered coefficients with %d ineqs is %f/%d" % (m, ave_rec_num, nb_of_unknowns))
    print("the success prob of recovering full ineqs with %d ineqs is %f" % (m, success_rate))

    return ave_rec_num, ave_rec_dis, success_rate


if __name__ == "__main__":
    k = 50
    with open("Data/Modular Hints/Mod_k/secret error/Kyber128/k50/es.txt", 'r') as g:
        solution = g.readlines()
    solution = np.array([int(x) for x in solution[0].split()])
    print("solution", solution)

    num_hints = []
    num_rec = []
    dis_rec = []
    suc_rat = []

    for m in tqdm(range(800, 801, 100)):
        num_hints.append(m)
        print("\nThe number of mod_q hints is", m)
        rec, dis, ratio = sol_mod_q_hints(m, k, 1, solution)
        num_rec.append(round(rec, 1))
        dis_rec.append(round(dis, 2))
        suc_rat.append(ratio)

    num_rec = [float(x) for x in num_rec]
    dis_rec = [float(x) for x in dis_rec]
    suc_rat = [float(x) for x in suc_rat]

    print("num_ine: ", num_hints)
    print("num_rec: ", num_rec)
    print("dis_rec: ", dis_rec)
    print("suc_rat: ", suc_rat)


