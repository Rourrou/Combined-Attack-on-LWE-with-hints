import numpy as np
from scipy.stats import binom, norm, uniform
import time
from proba import *
import math


def solve_inequalities_SMY4(eta, a, b, is_geq_zero, solution):
    # print("Solving inequalities...")

    [nb_of_inequalities, nb_of_unknowns] = a.shape

    # as>b
    for i in range(0, nb_of_inequalities):
        if is_geq_zero[i]:
            a[i] = np.array(a[i])
            b[i] = b[i]
        else:
            a[i] = np.array(-a[i])
            b[i] = -b[i]

    guess = np.zeros(nb_of_unknowns, dtype=int)  # creat an initial guess of the solution with all values set to zero

    if nb_of_inequalities == 0:
        return guess
    nb_of_values = 2 * eta + 1
    x = np.arange(-eta, eta + 1, dtype=np.int8)
    # x_pmf = binom.pmf(x + eta, 2 * eta, 0.5)
    x_pmf = uniform.cdf(x + 1, -eta, 2 * eta + 1) - uniform.cdf(x, -eta, 2 * eta + 1)
    # print("x_pmf", x_pmf)
    x_pmf = np.repeat(x_pmf.reshape(1, -1), nb_of_unknowns,
                      axis=0)  # this line repeats the x_pmf array multiple times to creat a 2D array
    a = a.astype(np.int16)  # this change the datatype of the matrix a to int16
    a_squared = np.square(a)  # this squares each element of a

    mean = np.matmul(x_pmf, x)  # 计算当前分布下，所有未知数的期望值
    variance = np.matmul(x_pmf, np.square(x)) - np.square(mean)  # 方差计算公式
    mean = np.multiply(a, np.repeat(mean[np.newaxis, :],
                                    nb_of_inequalities, axis=0))
    variance = np.multiply(
        a_squared,
        np.repeat(variance[np.newaxis, :], nb_of_inequalities, axis=0))
    # mean_all = mean.sum(axis=1).reshape(-1, 1).repeat(nb_of_unknowns, axis=1)
    mean = mean.sum(axis=1).reshape(-1, 1).repeat(nb_of_unknowns, axis=1) - mean
    # print("mean", mean)
    # mean -= b[:, np.newaxis]
    # mean_all += b[:, np.newaxis]
    mean += 800  # 实验表明，750， 800效果最佳
    # variance_all = variance.sum(axis=1).reshape(-1, 1).repeat(nb_of_unknowns, axis=1)
    variance = variance.sum(axis=1).reshape(-1, 1).repeat(nb_of_unknowns, axis=1) - variance
    # print("variance", variance)
    variance = np.clip(variance, 1, None)
    psuccess = np.zeros((nb_of_values, nb_of_inequalities,
                         nb_of_unknowns), dtype=float)
    for j in range(nb_of_values):
        zscore = np.divide(a * x[j] + mean + 0.5, np.sqrt(variance))
        # zscore = np.divide(a * x[j] + mean + 6*np.sqrt(variance), np.sqrt(variance))
        # print("norm.cdf(zscore)", norm.cdf(np.divide(100, np.sqrt(14500))))
        # psuccess[j, :, :] = norm.cdf(zscore) / norm.cdf(np.divide(mean + 0.5, np.sqrt(variance))) # central limit theorem
        psuccess[j, :, :] = norm.cdf(zscore)  # / norm.cdf(np.divide(mean_all, np.sqrt(variance_all)))
        # print("psuccess[j, 0, 0]", psuccess[j, 0, 0])
    psuccess = np.transpose(psuccess, axes=[2, 0, 1])
    psuccess = np.sum(np.log(psuccess), axis=2)
    # print("psuccess", psuccess)
    psuccess = np.exp(psuccess)
    # psuccess = np.sum(psuccess)

    x_pmf = np.multiply(psuccess, x_pmf)
    # print("x_pmf", x_pmf)
    row_sums = x_pmf.sum(axis=1)
    x_pmf /= row_sums[:, np.newaxis]

    # for i in range(3):
    #     print("x_pmf%d" %i)
    #     for element in x_pmf[i]:
    #         print(f"{element:.16f}")
    guess_ave = np.matmul(x_pmf, x)
    # print("guess_ave", guess_ave)

    max_pro = np.max(x_pmf, axis=1)
    # print("max_pro", max_pro)
    # print("max_pro_0,max_pro_1,max_pro_2,max_pro_3)", max_pro[0],max_pro[1], max_pro[2],max_pro[3])
    # 对数组进行降序排序
    sorted_indices = np.argsort(max_pro)[::-1]
    # print("sorted_indices", sorted_indices)
    # 选择前512个最大元素的索引
    index_pro = sorted_indices[:512]
    guess = x[np.argmax(x_pmf, axis=1)]
    # print("guess ", guess)

    if solution is not None:
        nb_correct = np.count_nonzero(solution == guess)
        # print("Number of correctly guessed unknowns: {:d}/{:d}".format(nb_correct, len(solution)))

    # 使用index_pro
    selected_guess = guess[index_pro]
    selected_solution = solution[index_pro]
    matches = selected_guess == selected_solution
    num_of_matches = np.count_nonzero(matches)
    # print("Number of selected coeffs matches:{:d}/512".format(num_of_matches))

    return guess, nb_correct


def solve_inequalities_DGJ19(eta, a, solution):
    # print("Solving inequalities...")
    qt = 3329 / 4
    [nb_of_inequalities, nb_of_unknowns] = a.shape
    # print("the nb_of_inequalities", nb_of_inequalities)
    guess = np.zeros(nb_of_unknowns, dtype=int)  # creat an initial guess of the solution with all values set to zero

    if nb_of_inequalities == 0:
        return guess
    nb_of_values = 2 * eta + 1
    x = np.arange(-eta, eta + 1, dtype=np.int8)
    x_pmf = binom.pmf(x + eta, 2 * eta, 0.5)
    # print("x_pmf", x_pmf)
    x_pmf = np.repeat(x_pmf.reshape(1, -1), nb_of_unknowns,
                      axis=0)  # this line repeats the x_pmf array multiple times to creat a 2D array
    a = a.astype(np.int16)  # this change the datatype of the matrix a to int16
    a_squared = np.square(a)  # this squares each element of a

    mean = np.matmul(x_pmf, x)  # 计算当前分布下，所有未知数的期望值
    variance = np.matmul(x_pmf, np.square(x)) - np.square(mean)  # 方差计算公式
    mean = np.multiply(a, np.repeat(mean[np.newaxis, :],
                                    nb_of_inequalities, axis=0))
    variance = np.multiply(
        a_squared,
        np.repeat(variance[np.newaxis, :], nb_of_inequalities, axis=0))
    # mean_all = mean.sum(axis=1).reshape(-1, 1).repeat(nb_of_unknowns, axis=1)
    mean = mean.sum(axis=1).reshape(-1, 1).repeat(nb_of_unknowns, axis=1) - mean
    # print("mean", mean)
    # mean -= b[:, np.newaxis]
    # mean_all += b[:, np.newaxis]
    # mean += qt # 实验表明，750， 800效果最佳
    variance = variance.sum(axis=1).reshape(-1, 1).repeat(nb_of_unknowns, axis=1) - variance
    # print("variance", variance)
    variance = np.clip(variance, 1, None)
    psuccess = np.zeros((nb_of_values, nb_of_inequalities,
                         nb_of_unknowns), dtype=float)
    for j in range(nb_of_values):
        zscore = np.divide(a * x[j] + mean + 0.5, np.sqrt(variance))
        # zscore = np.divide(a * x[j] + mean + 6*np.sqrt(variance), np.sqrt(variance))
        # print("norm.cdf(zscore)", norm.cdf(np.divide(100, np.sqrt(14500))))
        # psuccess[j, :, :] = norm.cdf(zscore) / norm.cdf(np.divide(mean + 0.5, np.sqrt(variance))) # central limit theorem
        psuccess[j, :, :] = norm.cdf(zscore)  # / norm.cdf(np.divide(mean_all, np.sqrt(variance_all)))
        # print("psuccess[j, 0, 0]", psuccess[j, 0, 0])
    psuccess = np.transpose(psuccess, axes=[2, 0, 1])
    psuccess = np.sum(np.log(psuccess), axis=2)
    # print("psuccess", psuccess)
    psuccess = np.exp(psuccess)
    # psuccess = np.sum(psuccess)

    x_pmf = np.multiply(psuccess, x_pmf)
    # print("x_pmf", x_pmf)
    row_sums = x_pmf.sum(axis=1)
    x_pmf /= row_sums[:, np.newaxis]

    # for i in range(3):
    #     print("x_pmf%d" %i)
    #     for element in x_pmf[i]:
    #         print(f"{element:.16f}")
    # guess_ave = np.matmul(x_pmf, x)
    # print("guess_ave", guess_ave)

    # max_pro = np.max(x_pmf, axis=1)
    # print("max_pro", max_pro)
    # print("max_pro_0,max_pro_1,max_pro_2,max_pro_3)", max_pro[0],max_pro[1], max_pro[2],max_pro[3])
    # 对数组进行降序排序
    # sorted_indices = np.argsort(max_pro)[::-1]
    # print("sorted_indices", sorted_indices)
    # 选择前512个最大元素的索引
    # index_pro = sorted_indices[:512]
    guess = x[np.argmax(x_pmf, axis=1)]
    # print("guess ", guess)

    if solution is not None:
        nb_correct = np.count_nonzero(solution == guess)
        print("Number of correctly guessed unknowns: {:d}/{:d}".format(nb_correct, len(solution)))

    # 使用index_pro
    # selected_guess = guess[index_pro]
    # selected_solution = solution[index_pro]
    # matches = selected_guess == selected_solution
    # num_of_matches = np.count_nonzero(matches)
    # print("Number of selected coeffs matches:{:d}/512".format(num_of_matches))

    return guess, nb_correct


def solve_inequalities_Del22_2(eta1, eta2, a, b, is_geq_zero, solution=None):
    eta1 = eta1
    eta2 = eta2
    m, n = a.shape

    s_list = []  # the list of secret s, e, r
    for i in range(0, 2 * eta1 + 1):
        s_list.append(i - eta1)
    s_dict = {}
    for i in range(0, 2 * eta1 + 1):
        s_dict[i - eta1] = 1 / (2 * eta1 + 1)
    # print("s_dict: ", s_dict)

    e1_list = []  # the list of secret s, e, r
    for i in range(0, 2 * eta2 + 1):
        e1_list.append(i - eta2)
    e1_dict = {}
    for i in range(0, 2 * eta2 + 1):
        e1_dict[i - eta2] = 1 / (2 * eta2 + 1)
    # print("e1_dict: ", e1_dict)

    e1_u = law_convolution(e1_dict, e1_dict)
    # print("e1_u: ", e1_u)

    # make the Conditional probability distribution
    er = law_product(s_dict, s_dict)
    # print("er: ", er)

    s_e1u = law_product(s_dict, e1_u)
    # print("s_e1u: ", s_e1u)

    tmp1 = iter_law_convolution(er, int(n / 2) - 1)  # !!tmp函数定义
    tmp2 = iter_law_convolution(s_e1u, int(n / 2) - 1)
    tmp = law_convolution(tmp1, tmp2)
    # tmp_full = law_convolution(iter_law_convolution(er, int(n/2)), iter_law_convolution(s_e1u, int(n/2)))
    # print(tmp)

    # generate random failures and calculate their angle with the secret

    failure = []
    b = b
    # as > b
    for i in range(0, m):
        if is_geq_zero[i]:
            df = np.array(a[i])
            b[i] = b[i]
        else:
            df = np.array(-a[i])
            b[i] = -b[i]
        failure.append(df)
    # print("b: ", b)

    sest_ave = [0] * n  # secret estimation using the original method of AGJ+19
    sest_lar = [0] * n  # secret estimation using the largest value

    tmp_e = law_convolution(tmp, s_e1u)
    tmp_e = tail_probability(tmp_e)
    # print("tmp_e", tmp_e)

    for i in range(0, int(n / 2)):
        fgivensci = []
        for j in range(0, 2 * eta1 + 1):
            pro_log = 0
            for k in range(0, m):
                p = 1 - tmp_e[b[k] + s_list[j] * failure[k][i]]
                # print("p", p)
                pro_log += np.log(p)
            fgivensci.append(pro_log)
            # print("pro_log", pro_log)

        # fgivensci = np.array(fgivensci) / sum(fgivensci)
        # print("fgivensci", fgivensci)
        max_log_prob = np.max(fgivensci)
        stable_fgivensci = fgivensci - max_log_prob
        # 将对数概率转换回普通概率
        fgivensci = np.exp(stable_fgivensci)

        # 归一化概率
        fgivensci = fgivensci / np.sum(fgivensci)
        # print(fgivensci)

        for j in range(0, 2 * eta1 + 1):
            # compute the secret key using the average value
            sest_ave[i] += (j - eta1) * fgivensci[j]
        # compute the secret key using the value with the largest prob
        sest_lar[i] = np.argmax(fgivensci) - eta1

    tmp_s = law_convolution(tmp, er)
    tmp_s = tail_probability(tmp_s)
    for i in range(int(n / 2), n):
        fgivensci = []
        for j in range(0, 2 * eta1 + 1):
            pro_log = 0
            for k in range(0, m):
                p = 1 - tmp_s[b[k] + s_list[j] * failure[k][i]]
                # print("p", p)
                pro_log += np.log(p)
            fgivensci.append(pro_log)
            # fgivensci.append(np.exp(pro_log)),会导致数值下溢
        # print("fgivensci", fgivensci)
        max_log_prob = np.max(fgivensci)
        stable_fgivensci = fgivensci - max_log_prob
        # 将对数概率转换回普通概率
        fgivensci = np.exp(stable_fgivensci)

        # 归一化概率
        fgivensci = fgivensci / np.sum(fgivensci)
        # print(fgivensci)

        for j in range(0, 2 * eta1 + 1):
            # compute the secret key using the average value
            sest_ave[i] += (j - eta1) * fgivensci[j]
        # compute the secret key using the value with the largest prob
        sest_lar[i] = np.argmax(fgivensci) - eta1

    E_int_ave = [0] * n
    for i in range(0, n):  # round numbers of secret
        E_int_ave[i] = round(sest_ave[i])
        if E_int_ave[i] > eta1:
            E_int_ave[i] = eta1
        elif E_int_ave[i] < -eta1:
            E_int_ave[i] = -eta1
    # print("sest_ave", E_int_ave)
    print("sest_lar", sest_lar)

    matnum_ave = 0
    matnum_lar = 0
    for i in range(0, n):  # round numbers of secret
        if E_int_ave[i] == solution[i]:
            matnum_ave += 1
        if sest_lar[i] == solution[i]:
            matnum_lar += 1

    print('DGJ+19_2, ciphertexts %d, experimental_matchnum using average value %d,'
          ' experimental_matchnum using largest prob value %d' % (m, matnum_ave, matnum_lar))


def solve_perfect_hints_Del22(eta, a, b, max_nb_of_iterations=10, solution=None, so_flag=None):
    if so_flag:
        print("Solving secret only perfect hints...")
    else:
        print("Solving secret error perfect hints...")

    eta = eta
    [nb_of_hints, nb_of_unknowns] = a.shape
    guess = np.zeros((nb_of_unknowns), dtype=int)  # creat an initial guess of the solution with all values set to zero
    if nb_of_hints == 0:
        return guess
    nb_of_values = 2 * eta + 1
    x = np.arange(-eta, eta + 1, dtype=np.int8)
    x_pmf = binom.pmf(x + eta, 2 * eta, 0.5)
    print("x_pmf: ", x_pmf)
    x_pmf = np.repeat(x_pmf.reshape(1, -1), nb_of_unknowns,
                      axis=0)  # this line repeats the x_pmf array multiple times to creat a 2D array

    a = a.astype(np.int16)  # this change the datatype of the matrix a to int16
    a_squared = np.square(a)  # this squares each element of a

    count = [0] * max_nb_of_iterations
    for z in range(max_nb_of_iterations):
        print("Iteration " + str(z))
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
            # zscore_pos = np.divide(a * x[j] + mean + 0.5 + 100, np.sqrt(variance))
            # zscore_neg = np.divide(a * x[j] + mean + 0.5 - 100, np.sqrt(variance))
            # psuccess[j, :, :] = norm.cdf(zscore_pos) - norm.cdf(zscore_neg)  # central limit theorem
            # SMY：连续分布无法计算d=zscore的概率，使用cdf(zscore+1)-cdf(zscore-1)
            zscore = np.divide(a * x[j] + mean + 0.5, np.sqrt(variance))
            # print("zscore", zscore)
            psuccess[j, :, :] = norm.cdf(zscore + 0.5) - norm.cdf(zscore - 0.5)  # Kyber128:0.5; Kyber256:1; Kyber512。
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
        print("Elapsed time: {:.1f} seconds".format(time_end - time_start))
        if solution is not None:
            nb_correct = np.count_nonzero(solution == guess)
            count[z] = nb_correct
            print("Number of correctly guessed unknowns: {:d}/{:d}"
                  .format(nb_correct, len(solution)))
        # if (z > 1) and count[z-1] >= count[z] + 1:
        #     # print(np.array(guess))
        #     count[z] = count[z - 1]
        #
        #     break
    print("count", count)
    print("guess", np.array(guess))
    short_vector = np.concatenate((np.array(guess - solution), np.array([1])))
    distance = np.linalg.norm(short_vector)
    distance = np.round(distance, 2)
    print("distance", distance)
    return guess, nb_correct, distance


def solve_ineq_perfect_hints_del22(eta, a, b, is_geq_zero, max_nb_of_iterations=20, solution=None):
    print("Solving translated secret only perfect hints...")

    [nb_of_hints, nb_of_unknowns] = a.shape
    guess = np.zeros(nb_of_unknowns, dtype=int)  # creat an initial guess of the solution with all values set to zero

    if nb_of_hints == 0:
        print("the number of hints is 0 !")
        return guess
    nb_of_values = 2 * eta + 1
    x = np.arange(-eta, eta + 1, dtype=np.int8)
    x_pmf = binom.pmf(x + eta, 2 * eta, 0.5)
    print("x_pmf", x_pmf)
    x_pmf = np.repeat(x_pmf.reshape(1, -1), nb_of_unknowns, axis=0)
    # x_pmf_static = x_pmf.copy()
    a = a.astype(np.int16)
    a_squared = np.square(a)

    count = [0] * max_nb_of_iterations
    for z in range(max_nb_of_iterations):
        print("Iteration " + str(z))
        print("**x_pmf", x_pmf[0, :])
        time_start = time.time()
        mean = np.matmul(x_pmf, x)  # 计算当前分布下，所有未知数的期望值
        variance = np.matmul(x_pmf, np.square(x)) - np.square(mean)
        mean = np.multiply(a, np.repeat(mean[np.newaxis, :], nb_of_hints, axis=0))
        variance = np.multiply(a_squared, np.repeat(variance[np.newaxis, :], nb_of_hints, axis=0))
        mean = mean.sum(axis=1).reshape(-1, 1).repeat(nb_of_unknowns, axis=1) - mean  # 减去自身
        print("mean", mean[:3, :3])
        mean -= b[:, np.newaxis]
        print("mean", mean[:3, :3])
        variance = variance.sum(axis=1).reshape(-1, 1).repeat(nb_of_unknowns, axis=1) - variance
        variance = np.clip(variance, 1, None)
        psuccess = np.zeros((nb_of_values, nb_of_hints, nb_of_unknowns), dtype=float)
        for j in range(nb_of_values):
            zscore = np.divide(a * x[j] + mean + 0.5, np.sqrt(variance))
            psuccess[j, :, :] = norm.cdf(zscore)  # central limit theorem
        print("psuccess[, 0, :5]", psuccess[:, 0, :5])

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
        print("x_pmf", x_pmf[:5, :])
        guess = x[np.argmax(x_pmf, axis=1)]
        print("guess", np.array(guess))

        if z == max_nb_of_iterations - 1:
            print(np.array(guess))

        time_end = time.time()
        print("Elapsed time: {:.1f} seconds".format(time_end - time_start))

        if solution is not None:
            nb_correct = np.count_nonzero(solution == guess)
            count[z] = nb_correct
            print("Number of correctly guessed unknowns: {:d}/{:d}".format(nb_correct, len(solution)))

        if (z > 1) and count[z - 1] >= count[z] + 2:
            # print(np.array(guess))
            count[z] = count[z - 1]
            break
    print("count", count)
    print("guess", np.array(guess))
    short_vector = np.concatenate((np.array(guess - solution), np.array([1])))
    distance = np.linalg.norm(short_vector)
    distance = np.round(distance, 2)
    print("distance", distance)
    return guess, nb_correct, distance


def solve_approx_hints_Del22(eta, sigma, V, L, max_nb_of_iterations=10, solution=None, so_flag=None):
    if so_flag:
        print("Solving secret only approximate hints...")
    else:
        print("Solving secret error approximate hints...")

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

        if z == max_nb_of_iterations - 1:
            print(np.array(guess))

        time_end = time.time()
        print("Elapsed time: {:.1f} seconds".format(time_end - time_start))
        if solution is not None:
            nb_correct = np.count_nonzero(solution == guess)
            count[z] = nb_correct
            print("Number of correctly guessed unknowns: {:d}/{:d}"
                  .format(nb_correct, len(solution)))
        # if (z > 1) and count[z-1] >= count[z] + 1:
        #     # print(np.array(guess))
        #     count[z] = count[z - 1]
        #
        #     break

    print("count", count)
    # print("guess", np.array(guess))
    short_vector = np.concatenate((np.array(guess - solution), np.array([1])))
    distance = np.linalg.norm(short_vector)
    distance = np.round(distance, 2)
    # print("distance", distance)
    return guess, nb_correct, distance
