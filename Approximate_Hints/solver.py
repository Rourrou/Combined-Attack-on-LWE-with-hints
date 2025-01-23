# from kyber import KyberPKE, KyberKEM
# from rng import RNG_OS
# import numpy as np
# from scipy.stats import binom, norm, uniform
# import time
# from proba import *
# import math
#
# np.set_printoptions(linewidth=np.inf)
#
#
# def poly_multiplier_to_matrix(poly, row_index=None):
#     # Transform a polynomial representaion into its matrix form
#     p = poly.astype(np.int16)
#     if row_index is None:
#         N = len(p)
#         r = np.zeros((N, N), dtype=np.int16)
#         for i in range(N):
#             r[i, :] = np.concatenate((np.flip(p[:i + 1]), -np.flip(p[i + 1:])))
#     else:
#         r = np.concatenate((np.flip(p[:row_index + 1]),
#                             -np.flip(p[row_index + 1:])))
#     return r
#
#
# def manipulate_ciphertext(pke, c, index):
#     c2 = np.copy(c)
#     nb_of_bytes = KyberPKE.N * pke.DV // 8
#     v = KyberPKE.decode(c[-nb_of_bytes:], pke.DV, KyberPKE.N)
#     v_mod = (v[index] + 2 ** (pke.DV - 2)) % (2 ** (pke.DV))
#     error = KyberPKE.decompress(v_mod, pke.DV) \
#             - KyberPKE.decompress(v[index], pke.DV)
#     error %= KyberPKE.Q
#     v[index] = v_mod
#     v = KyberPKE.encode(v, pke.DV)
#     c2[-nb_of_bytes:] = v
#     return [c2, error]
#
#
# def generate_inequalities(
#         kem,
#         public_key,
#         nb_of_inequalities,
#         index=None,
#         bias_threshold=None,
#         max_nb_of_encapsulations=None,
#         return_manipulation=False,
#         verbose=True):
#     if verbose:
#         print("Generating {:d} inequalities for {:s}..."
#               .format(nb_of_inequalities, kem.version()))
#     (K, N) = (kem.pke.K, KyberPKE.N)
#     a = np.zeros((nb_of_inequalities, 2 * K * N), dtype=np.int16)
#     b = np.zeros((nb_of_inequalities), dtype=np.int16)
#     if return_manipulation:
#         manipulated_indices = np.zeros(nb_of_inequalities, dtype=np.uint8)
#         ciphertexts = np.zeros((nb_of_inequalities,
#                                 (K * kem.pke.DU + kem.pke.DV) * N // 8), dtype=np.uint8)
#         manipulated_ciphertexts = np.zeros((nb_of_inequalities,
#                                             (K * kem.pke.DU + kem.pke.DV) * N // 8), dtype=np.uint8)
#         shared_secrets = np.zeros((nb_of_inequalities,
#                                    KyberKEM.SHARED_SECRET_BYTES), dtype=np.uint8)
#         manipulated_shared_secrets = np.zeros((nb_of_inequalities,
#                                                KyberKEM.SHARED_SECRET_BYTES), dtype=np.uint8)
#     for i in range(nb_of_inequalities):
#         done = False
#         nb_of_encapsulations = 0
#         lowest_bias = 10000
#         while not done:
#             c, ss, m, r, e1, du, e2, dv, ss_pre = kem.encapsulate(public_key,
#                                                                   return_internals=True)
#             b2 = e2 + dv
#             m = np.unpackbits(m, bitorder='little')
#             _, error = manipulate_ciphertext(kem.pke, c, np.arange(N))
#             b2[np.logical_and(error == 832, m == 0)] -= 1  # 832 = floor(prime/4)
#             b2[np.logical_and(error == 833, m == 1)] += 1
#             bias = abs(b2)
#             ind = np.argmin(bias) if index is None else index
#             if return_manipulation:
#                 manipulated_indices[i] = ind
#                 ciphertexts[i, :] = c
#                 manipulated_ciphertexts[i, :], _ = \
#                     manipulate_ciphertext(kem.pke, c, ind)
#                 shared_secrets[i, :] = ss
#                 manipulated_shared_secrets[i, :] = KyberKEM.KDF(
#                     np.concatenate((ss_pre,
#                                     KyberKEM.H(manipulated_ciphertexts[i, :]))))
#             if bias[ind] < lowest_bias:
#                 lowest_bias = bias[ind]
#                 b[i] = b2[ind]
#                 for j in range(K):
#                     a[i, j * N:(j + 1) * N] = poly_multiplier_to_matrix(
#                         -e1[j, :] - du[j, :], row_index=ind)
#                     a[i, (K + j) * N:(K + j + 1) * N] = poly_multiplier_to_matrix(
#                         r[j, :], row_index=ind)
#             nb_of_encapsulations += 1
#             done = (bias_threshold is None) \
#                    or (bias[ind] <= bias_threshold) \
#                    or ((max_nb_of_encapsulations is not None) and \
#                        (nb_of_encapsulations >= max_nb_of_encapsulations))
#     return [a, b, manipulated_indices, ciphertexts, manipulated_ciphertexts, \
#             shared_secrets, manipulated_shared_secrets] if return_manipulation \
#         else [a, b]
#
#
# def evaluate_inequalities_slow(
#         kem,
#         private_key,
#         manipulated_indices,
#         manipulated_ciphertexts,
#         manipulated_shared_secrets):
#     nb_of_inequalities = manipulated_ciphertexts.shape[0]
#     is_geq_zero = np.full((nb_of_inequalities), False)
#     for i in range(nb_of_inequalities):
#         ss = kem.decapsulate(private_key, manipulated_ciphertexts[i, :],
#                              roulette_index=manipulated_indices[i])
#         is_geq_zero[i] = not np.array_equal(ss,
#                                             manipulated_shared_secrets[i, :])
#     return is_geq_zero
#
#
# def evaluate_inequalities_fast(a, b, solution):  # evaluate the direction of inequalities
#     # print(np.matmul(a, solution))
#     return (np.matmul(a, solution) + b) >= 0
#
#
# def evaluate_inequalities_fast_2(a, b, solution):  # evaluate the direction of inequalities
#     # print(np.matmul(a, solution))
#     return (np.matmul(a, solution) - b) >= 0
#
#
# def corrupt_inequalities(is_geq_zero, prob_success_is_missed, verbose=True):
#     # introduces corruption to the inequalities based on a given probility
#     is_geq_zero_corrupt = np.copy(is_geq_zero)
#     ind, = np.where(is_geq_zero == False)
#     miss = np.random.binomial(1, prob_success_is_missed, size=len(ind))
#     ind2, = np.where(miss == 1)
#     is_geq_zero_corrupt[ind[ind2]] = True
#     print("Corrupted {:d} out of {:d} inequalities"
#           .format(len(ind2), len(is_geq_zero)))
#     return is_geq_zero_corrupt
#
#
# def generate_equalities(kem, public_key, verbose=True):
#     if verbose:
#         print("Generating equalities for {:s}...".format(kem.version()))
#     (K, N) = (kem.pke.K, KyberPKE.N)
#     (t, rho) = np.split(public_key, [K * N * 3 // 2])
#     t = np.reshape(KyberPKE.decode(t, 12, K * N), (K, N))
#     A = kem.pke.generate_A(rho, transpose=False)
#     a = np.hstack((np.zeros((K * N, K * N), dtype=np.int32),
#                    2285 * np.eye(K * N, dtype=np.int32)))
#     b = np.zeros((K * N), dtype=np.int32)
#     for i in range(K):
#         b[i * N:(i + 1) * N] = -kem.pke.INTT(t[i])
#         for j in range(K):
#             a[i * N:(i + 1) * N, j * N:(j + 1) * N] = poly_multiplier_to_matrix(
#                 kem.pke.INTT(A[i, j]))
#     return a, b
#
#
# def solve_inequalities(kem, a, b, is_geq_zero,
#                        max_nb_of_iterations=10,
#                        verbose=True,
#                        solution=None):  # analyze convergence rate with a known solution
#     if verbose:
#         print("Solving inequalities...")
#     eta = kem.pke.ETA1
#     [nb_of_inequalities, nb_of_unknowns] = a.shape
#     guess = np.zeros((nb_of_unknowns), dtype=int)  # creat an intial guess of the solution with all values set to zero
#     if verbose and solution is not None:
#         nb_correct = np.count_nonzero(solution == guess)
#         print("Number of correctly guessed unknowns: {:d}/{:d}"
#               .format(nb_correct, len(solution)))
#     if nb_of_inequalities == 0:
#         return guess
#     # 计算概率质量函数
#     nb_of_values = 2 * eta + 1
#     x = np.arange(-eta, eta + 1, dtype=np.int8)
#     x_pmf = binom.pmf(x + eta, 2 * eta,
#                       0.5)  # x_pmf will be an array containing the binomial probilities corresponding to each value in the shifted x
#     x_pmf = np.repeat(x_pmf.reshape(1, -1), nb_of_unknowns,
#                       axis=0)  # this line repeats the x_pmf array multiple times to creat a 2D array
#
#     # 转换a的数据类型以进行计算
#     a = a.astype(np.int16)  # this change the datatype of the matrix a to int16
#     a_squared = np.square(a)  # this squares each element of a
#     prob_geq_zero = np.zeros((nb_of_inequalities), dtype=float)  # P_fail[i]
#
#     # 计算观察到的失败的概率
#     p_failure_is_observed = np.count_nonzero(is_geq_zero) / nb_of_inequalities  # As+b>0的统计概率
#
#     # 使用中心极限定理计算实际的失败概率
#     mean = np.matmul(x_pmf, x)
#     variance = np.matmul(x_pmf, np.square(x)) - np.square(mean)
#     mean = np.matmul(a, mean)  # adjust the mean and variance using matrix multiplication
#     variance = np.matmul(a_squared, variance)
#     zscore = np.divide(mean + 0.5 + b, np.sqrt(variance))
#     p_failure_is_reality = norm.cdf(zscore)  # central limit theorem Eq(15)
#     p_failure_is_reality = np.mean(p_failure_is_reality)
#
#     # 计算不等式是否正确的概率
#     p_inequality_is_correct = min(
#         p_failure_is_reality / p_failure_is_observed, 1.0)
#
#     prob_geq_zero[is_geq_zero] = p_inequality_is_correct
#
#     # 初始化适应度
#     fitness = np.zeros((max_nb_of_iterations), dtype=float)
#     fitness_max = np.sum(np.maximum(prob_geq_zero, 1 - prob_geq_zero))
#
#     # 进行多次迭代，优化解决不等式的解
#     for z in range(max_nb_of_iterations):
#         if verbose:
#             print("Iteration " + str(z))
#             time_start = time.time()
#         mean = np.matmul(x_pmf, x)
#         variance = np.matmul(x_pmf, np.square(x)) - np.square(mean)
#         mean = np.multiply(a, np.repeat(mean[np.newaxis, :],
#                                         nb_of_inequalities, axis=0))
#         variance = np.multiply(
#             a_squared,
#             np.repeat(variance[np.newaxis, :], nb_of_inequalities, axis=0))
#         mean = mean.sum(axis=1).reshape(-1, 1).repeat(nb_of_unknowns, axis=1) \
#                - mean
#         mean += b[:, np.newaxis]
#         variance = variance.sum(axis=1).reshape(-1, 1).repeat(nb_of_unknowns,
#                                                               axis=1) - variance
#         variance = np.clip(variance, 1, None)
#         psuccess = np.zeros((nb_of_values, nb_of_inequalities,
#                              nb_of_unknowns), dtype=float)
#         for j in range(nb_of_values):
#             zscore = np.divide(a * x[j] + mean + 0.5, np.sqrt(variance))
#             psuccess[j, :, :] = norm.cdf(zscore)  # central limit theorem
#         psuccess = np.transpose(psuccess, axes=[2, 0, 1])
#         psuccess = \
#             np.multiply(psuccess, prob_geq_zero[np.newaxis, np.newaxis, :]) + \
#             np.multiply(1 - psuccess, 1 - prob_geq_zero[np.newaxis, np.newaxis, :])
#         psuccess = np.clip(psuccess, 10e-5, None)
#         psuccess = np.sum(np.log(psuccess), axis=2)
#         row_means = psuccess.max(axis=1)
#         psuccess -= row_means[:, np.newaxis]
#         psuccess = np.exp(psuccess)
#         x_pmf = np.multiply(psuccess, x_pmf)
#         row_sums = x_pmf.sum(axis=1)
#         x_pmf /= row_sums[:, np.newaxis]
#
#         guess = x[np.argmax(x_pmf, axis=1)]
#         fit = (np.matmul(a, guess) + b >= 0).astype(float)
#         fit = np.dot(fit, prob_geq_zero) + np.dot(1 - fit, 1 - prob_geq_zero)
#         fitness[z] = fit / fitness_max
#         if verbose:
#             time_end = time.time()
#             print("Elapsed time: {:.1f} seconds".format(time_end - time_start))
#             print("Fitness {:.2f}%".format(fitness[z] * 100))
#             if solution is not None:
#                 nb_correct = np.count_nonzero(solution == guess)
#                 print("Number of correctly guessed unknowns: {:d}/{:d}"
#                       .format(nb_correct, len(solution)))
#         if (z > 1) and fitness[z - 1] >= fitness[z]:
#             break
#     return guess
#
#
# # 假设不等式成功概率为1
# def solve_inequalities_with_prob1(kem, a, b, is_geq_zero,
#                                   max_nb_of_iterations=10,
#                                   verbose=True,
#                                   solution=None):  # analyze convergence rate with a known solution
#     if verbose:
#         print("Solving inequalities...")
#     eta = kem.pke.ETA1
#     [nb_of_inequalities, nb_of_unknowns] = a.shape
#     guess = np.zeros((nb_of_unknowns), dtype=int)  # creat an intial guess of the solution with all values set to zero
#     if verbose and solution is not None:
#         nb_correct = np.count_nonzero(solution == guess)
#         print("Number of correctly guessed unknowns: {:d}/{:d}"
#               .format(nb_correct, len(solution)))
#     if nb_of_inequalities == 0:
#         return guess
#     # 计算概率质量函数
#     nb_of_values = 2 * eta + 1
#     x = np.arange(-eta, eta + 1, dtype=np.int8)
#     x_pmf = binom.pmf(x + eta, 2 * eta,
#                       0.5)  # x_pmf will be an array containing the binomial probilities corresponding to each value in the shifted x
#     x_pmf = np.repeat(x_pmf.reshape(1, -1), nb_of_unknowns,
#                       axis=0)  # this line repeats the x_pmf array multiple times to creat a 2D array
#
#     # 转换a的数据类型以进行计算
#     a = a.astype(np.int16)  # this change the datatype of the matrix a to int16
#     a_squared = np.square(a)  # this squares each element of a
#     prob_geq_zero = np.zeros((nb_of_inequalities), dtype=float)  # P_fail[i]
#
#     # 计算观察到的失败的概率
#     p_failure_is_observed = np.count_nonzero(is_geq_zero) / nb_of_inequalities  # As+b>0的统计概率
#
#     # 使用中心极限定理计算实际的失败概率
#     mean = np.matmul(x_pmf, x)
#     variance = np.matmul(x_pmf, np.square(x)) - np.square(mean)
#     mean = np.matmul(a, mean)  # adjust the mean and variance using matrix multiplication
#     variance = np.matmul(a_squared, variance)
#     zscore = np.divide(mean + 0.5 + b, np.sqrt(variance))
#     p_failure_is_reality = norm.cdf(zscore)  # central limit theorem Eq(15)
#     p_failure_is_reality = np.mean(p_failure_is_reality)
#
#     # 计算不等式是否正确的概率
#     p_inequality_is_correct = min(
#         p_failure_is_reality / p_failure_is_observed, 1.0)
#
#     prob_geq_zero[is_geq_zero] = p_inequality_is_correct
#
#     # 初始化适应度
#     fitness = np.zeros((max_nb_of_iterations), dtype=float)
#     fitness_max = np.sum(np.maximum(prob_geq_zero, 1 - prob_geq_zero))
#
#     # 进行多次迭代，优化解决不等式的解
#     for z in range(max_nb_of_iterations):
#         if verbose:
#             print("Iteration " + str(z))
#             time_start = time.time()
#         mean = np.matmul(x_pmf, x)
#         variance = np.matmul(x_pmf, np.square(x)) - np.square(mean)
#         mean = np.multiply(a, np.repeat(mean[np.newaxis, :],
#                                         nb_of_inequalities, axis=0))
#         variance = np.multiply(
#             a_squared,
#             np.repeat(variance[np.newaxis, :], nb_of_inequalities, axis=0))
#         mean = mean.sum(axis=1).reshape(-1, 1).repeat(nb_of_unknowns, axis=1) \
#                - mean
#         mean += b[:, np.newaxis]
#         variance = variance.sum(axis=1).reshape(-1, 1).repeat(nb_of_unknowns,
#                                                               axis=1) - variance
#         variance = np.clip(variance, 1, None)
#         psuccess = np.zeros((nb_of_values, nb_of_inequalities,
#                              nb_of_unknowns), dtype=float)
#         for j in range(nb_of_values):
#             zscore = np.divide(a * x[j] + mean + 0.5, np.sqrt(variance))
#             psuccess[j, :, :] = norm.cdf(zscore)  # central limit theorem
#         psuccess = np.transpose(psuccess, axes=[2, 0, 1])
#         psuccess = \
#             np.multiply(psuccess, prob_geq_zero[np.newaxis, np.newaxis, :]) + \
#             np.multiply(1 - psuccess, 1 - prob_geq_zero[np.newaxis, np.newaxis, :])
#         psuccess = np.clip(psuccess, 10e-5, None)
#         psuccess = np.sum(np.log(psuccess), axis=2)
#         row_means = psuccess.max(axis=1)
#         psuccess -= row_means[:, np.newaxis]
#         psuccess = np.exp(psuccess)
#         x_pmf = np.multiply(psuccess, x_pmf)
#         row_sums = x_pmf.sum(axis=1)
#         x_pmf /= row_sums[:, np.newaxis]
#
#         guess = x[np.argmax(x_pmf, axis=1)]
#         fit = (np.matmul(a, guess) + b >= 0).astype(float)
#         fit = np.dot(fit, prob_geq_zero) + np.dot(1 - fit, 1 - prob_geq_zero)
#         fitness[z] = fit / fitness_max
#         if verbose:
#             time_end = time.time()
#             print("Elapsed time: {:.1f} seconds".format(time_end - time_start))
#             print("Fitness {:.2f}%".format(fitness[z] * 100))
#             if solution is not None:
#                 nb_correct = np.count_nonzero(solution == guess)
#                 print("Number of correctly guessed unknowns: {:d}/{:d}"
#                       .format(nb_correct, len(solution)))
#         if (z > 1) and fitness[z - 1] >= fitness[z]:
#             break
#     return guess
#
#
# # 分组BP算法，每k个私钥一个block
# # block = 2
# def solve_inequalities_block(kem, a, b, is_geq_zero,
#                              max_nb_of_iterations=10,
#                              verbose=True,
#                              solution=None):  # analyze convergence rate with a known solution
#     if verbose:
#         print("Solving inequalities...")
#     eta = kem.pke.ETA1
#     [nb_of_inequalities, nb_of_unknowns] = a.shape
#     guess = np.zeros((nb_of_unknowns), dtype=int)  # creat an intial guess of the solution with all values set to zero
#     if verbose and solution is not None:
#         nb_correct = np.count_nonzero(solution == guess)
#         print("Number of correctly guessed unknowns: {:d}/{:d}"
#               .format(nb_correct, len(solution)))
#     if nb_of_inequalities == 0:
#         return guess
#     # 计算概率质量函数
#     nb_of_values = 2 * eta + 1
#     x = np.arange(-eta, eta + 1, dtype=np.int8)
#     x_pmf = binom.pmf(x + eta, 2 * eta,
#                       0.5)  # x_pmf will be an array containing the binomial probilities corresponding to each value in the shifted x
#     print("x_pmf", x_pmf)
#     x_pmf_block = np.outer(x_pmf, x_pmf)
#     print("x_pmf_block", x_pmf_block)
#     x_pmf = np.repeat(x_pmf.reshape(1, -1), nb_of_unknowns,
#                       axis=0)  # this line repeats the x_pmf array multiple times to creat a 2D array
#
#     # 转换a的数据类型以进行计算
#     a = a.astype(np.int16)  # this change the datatype of the matrix a to int16
#     a_squared = np.square(a)  # this squares each element of a
#     prob_geq_zero = np.zeros((nb_of_inequalities), dtype=float)  # P_fail[i]
#
#     # 计算观察到的失败的概率
#     p_failure_is_observed = np.count_nonzero(is_geq_zero) / nb_of_inequalities  # As+b>0的统计概率
#
#     # 使用中心极限定理计算实际的失败概率
#     mean = np.matmul(x_pmf, x)
#     variance = np.matmul(x_pmf, np.square(x)) - np.square(mean)
#     mean = np.matmul(a, mean)  # adjust the mean and variance using matrix multiplication
#     variance = np.matmul(a_squared, variance)
#     zscore = np.divide(mean + 0.5 + b, np.sqrt(variance))
#     p_failure_is_reality = norm.cdf(zscore)  # central limit theorem Eq(15)
#     p_failure_is_reality = np.mean(p_failure_is_reality)
#
#     # 计算不等式是否正确的概率
#     p_inequality_is_correct = min(
#         p_failure_is_reality / p_failure_is_observed, 1.0)
#
#     prob_geq_zero[is_geq_zero] = p_inequality_is_correct
#
#     # 初始化适应度
#     fitness = np.zeros((max_nb_of_iterations), dtype=float)
#     fitness_max = np.sum(np.maximum(prob_geq_zero, 1 - prob_geq_zero))
#
#     # 进行多次迭代，优化解决不等式的解
#     for z in range(max_nb_of_iterations):
#         if verbose:
#             print("Iteration " + str(z))
#             time_start = time.time()
#         mean = np.matmul(x_pmf, x)
#         variance = np.matmul(x_pmf, np.square(x)) - np.square(mean)
#         mean = np.multiply(a, np.repeat(mean[np.newaxis, :],
#                                         nb_of_inequalities, axis=0))
#         print("mean", mean)
#         variance = np.multiply(
#             a_squared,
#             np.repeat(variance[np.newaxis, :], nb_of_inequalities, axis=0))
#
#         mean_block = np.array([mean[:, i] + mean[:, i + 1] for i in range(0, nb_of_unknowns, 2)]).T
#         print("mean_block", mean_block)
#
#         variance_block = np.array([variance[:, i] + variance[:, i + 1] for i in range(0, nb_of_unknowns, 2)]).T
#         print("variance_block", variance_block)
#
#         mean = mean.sum(axis=1).reshape(-1, 1).repeat(int(nb_of_unknowns / 2), axis=1) - mean_block
#         print("mean", mean)
#
#         mean += b[:, np.newaxis]
#         print("mean+b", mean)
#
#         variance = variance.sum(axis=1).reshape(-1, 1).repeat(nb_of_unknowns / 2, axis=1) - variance_block
#
#         variance = np.clip(variance, 1, None)
#         psuccess = np.zeros((nb_of_values, nb_of_inequalities,
#                              nb_of_unknowns), dtype=float)
#
#         for i in range(nb_of_values):
#             for j in range(nb_of_values):
#                 zscore = a * x[j] + mean + 0.5
#                 zscore = np.divide(zscore, np.sqrt(variance))
#                 psuccess[j, :, :] = norm.cdf(zscore)  # central limit theorem
#
#         psuccess = np.transpose(psuccess, axes=[2, 0, 1])
#         psuccess = \
#             np.multiply(psuccess, prob_geq_zero[np.newaxis, np.newaxis, :]) + \
#             np.multiply(1 - psuccess, 1 - prob_geq_zero[np.newaxis, np.newaxis, :])
#         psuccess = np.clip(psuccess, 10e-5, None)
#         psuccess = np.sum(np.log(psuccess), axis=2)
#         row_means = psuccess.max(axis=1)
#         psuccess -= row_means[:, np.newaxis]
#         psuccess = np.exp(psuccess)
#         x_pmf = np.multiply(psuccess, x_pmf)
#         row_sums = x_pmf.sum(axis=1)
#         x_pmf /= row_sums[:, np.newaxis]
#
#         guess = x[np.argmax(x_pmf, axis=1)]
#         fit = (np.matmul(a, guess) + b >= 0).astype(float)
#         fit = np.dot(fit, prob_geq_zero) + np.dot(1 - fit, 1 - prob_geq_zero)
#         fitness[z] = fit / fitness_max
#         if verbose:
#             time_end = time.time()
#             print("Elapsed time: {:.1f} seconds".format(time_end - time_start))
#             print("Fitness {:.2f}%".format(fitness[z] * 100))
#             if solution is not None:
#                 nb_correct = np.count_nonzero(solution == guess)
#                 print("Number of correctly guessed unknowns: {:d}/{:d}"
#                       .format(nb_correct, len(solution)))
#         if (z > 1) and fitness[z - 1] >= fitness[z]:
#             break
#     return guess
#
#
# # change the code to data from BCD
# def solve_inequalities_SMY(eta, a, b, is_geq_zero,
#                            max_nb_of_iterations=5,
#                            verbose=True,
#                            solution=None):  # analyze convergence rate with a known solution
#     if verbose:
#         print("Solving inequalities...")
#     eta = eta
#     [nb_of_inequalities, nb_of_unknowns] = a.shape
#     guess = np.zeros((nb_of_unknowns), dtype=int)  # creat an initial guess of the solution with all values set to zero
#     if verbose and solution is not None:
#         nb_correct = np.count_nonzero(solution == guess)
#         print("Number of correctly guessed unknowns: {:d}/{:d}"
#               .format(nb_correct, len(solution)))
#     if nb_of_inequalities == 0:
#         return guess
#     nb_of_values = 2 * eta + 1
#     x = np.arange(-eta, eta + 1, dtype=np.int8)
#     x_pmf = binom.pmf(x + eta, 2 * eta, 0.5)
#     print(x_pmf)
#     x_pmf = np.repeat(x_pmf.reshape(1, -1), nb_of_unknowns,
#                       axis=0)  # this line repeats the x_pmf array multiple times to creat a 2D array
#     # print(x_pmf[1])
#     a = a.astype(np.int16)  # this change the datatype of the matrix a to int16
#     a_squared = np.square(a)  # this squares each element of a
#
#     prob_geq_zero = np.zeros((nb_of_inequalities), dtype=float)
#     p_failure_is_observed = np.count_nonzero(is_geq_zero) / nb_of_inequalities
#     print("p_failure_is_observed", p_failure_is_observed)
#     mean = np.matmul(x_pmf, x)
#     variance = np.matmul(x_pmf, np.square(x)) - np.square(mean)
#     mean = np.matmul(a, mean)  # adjust the mean and variance using matrix multiplication
#     variance = np.matmul(a_squared, variance)
#     zscore = np.divide(mean + 0.5 - b, np.sqrt(variance))
#     p_failure_is_reality = norm.cdf(zscore)  # central limit theorem
#     p_failure_is_reality = np.mean(p_failure_is_reality)
#     print("p_failure_is_reality", p_failure_is_reality)
#     p_inequality_is_correct = min(
#         p_failure_is_reality / p_failure_is_observed, 1.0)
#     print("p_inequality_is_correct", p_inequality_is_correct)
#     prob_geq_zero[is_geq_zero] = p_inequality_is_correct
#     # print("prob_geq_zero",prob_geq_zero)
#     fitness = np.zeros((max_nb_of_iterations), dtype=float)
#     fitness_max = np.sum(np.maximum(prob_geq_zero, 1 - prob_geq_zero))  # 正确>0的个数
#     # print(guess[1])
#     count = [0] * max_nb_of_iterations
#     for z in range(max_nb_of_iterations):
#         if verbose:
#             print("Iteration " + str(z))
#             time_start = time.time()
#         mean = np.matmul(x_pmf, x)  # 计算当前分布下，所有未知数的期望值
#         variance = np.matmul(x_pmf, np.square(x)) - np.square(mean)
#         mean = np.multiply(a, np.repeat(mean[np.newaxis, :],
#                                         nb_of_inequalities, axis=0))
#         variance = np.multiply(
#             a_squared,
#             np.repeat(variance[np.newaxis, :], nb_of_inequalities, axis=0))
#         mean = mean.sum(axis=1).reshape(-1, 1).repeat(nb_of_unknowns, axis=1) \
#                - mean
#         mean += b[:, np.newaxis]
#         variance = variance.sum(axis=1).reshape(-1, 1).repeat(nb_of_unknowns,
#                                                               axis=1) - variance
#         variance = np.clip(variance, 1, None)
#         psuccess = np.zeros((nb_of_values, nb_of_inequalities,
#                              nb_of_unknowns), dtype=float)
#         for j in range(nb_of_values):
#             zscore = np.divide(a * x[j] + mean + 0.5, np.sqrt(variance))
#             psuccess[j, :, :] = norm.cdf(zscore)  # central limit theorem
#         psuccess = np.transpose(psuccess, axes=[2, 0, 1])
#         psuccess = \
#             np.multiply(psuccess, prob_geq_zero[np.newaxis, np.newaxis, :]) + \
#             np.multiply(1 - psuccess, 1 - prob_geq_zero[np.newaxis, np.newaxis, :])
#         psuccess = np.clip(psuccess, 10e-5, None)
#         # psuccess = np.clip(psuccess, 10e-15, None)
#         psuccess = np.sum(np.log(psuccess), axis=2)
#         row_means = psuccess.max(axis=1)
#         psuccess -= row_means[:, np.newaxis]
#         psuccess = np.exp(psuccess)
#
#         x_pmf = np.multiply(psuccess, x_pmf)
#         row_sums = x_pmf.sum(axis=1)
#         x_pmf /= row_sums[:, np.newaxis]
#         print(x_pmf[:3])
#         max_pro = np.max(x_pmf, axis=1)
#         # print(max_pro)S
#         # 对数组进行降序排序
#         sorted_indices = np.argsort(max_pro)[::-1]
#         # 选择前512个最大元素的索引
#         index_pro = sorted_indices[:512]
#         # print(index_pro)
#         guess = x[np.argmax(x_pmf, axis=1)]
#         fit = (np.matmul(a, guess) + b >= 0).astype(float)
#         fit = np.dot(fit, prob_geq_zero) + np.dot(1 - fit, 1 - prob_geq_zero)
#         fitness[z] = fit / fitness_max
#         if z == max_nb_of_iterations - 1:
#             print(np.array(guess))
#         if verbose:
#             time_end = time.time()
#             print("Elapsed time: {:.1f} seconds".format(time_end - time_start))
#             print("Fitness {:.2f}%".format(fitness[z] * 100))
#             if solution is not None:
#                 nb_correct = np.count_nonzero(solution == guess)
#                 count[z] = nb_correct
#                 print("Number of correctly guessed unknowns: {:d}/{:d}"
#                       .format(nb_correct, len(solution)))
#         # 使用index_pro
#         selected_guess = guess[index_pro]
#         selected_solution = solution[index_pro]
#         matches = selected_guess == selected_solution
#         num_of_matches = np.count_nonzero(matches)
#         print("Number of selected coeffs matches:{:d}/512"
#               .format(num_of_matches))
#         if (z > 1) and count[z - 1] >= count[z] + 1:
#             print(np.array(guess))
#             break
#
#     return guess
#
#
# # change the code to data from uniform
# def solve_inequalities_SMY_u(eta, a, b, is_geq_zero,
#                              max_nb_of_iterations=5,
#                              verbose=True,
#                              solution=None):  # analyze convergence rate with a known solution
#     if verbose:
#         print("Solving inequalities...")
#     eta = eta
#     [nb_of_inequalities, nb_of_unknowns] = a.shape
#     for i in range(0, nb_of_inequalities):
#         # as+b > 0
#         if is_geq_zero[i]:
#             a[i] = np.array(a[i])
#             b[i] = -b[i]
#         else:
#             a[i] = np.array(-a[i])
#             b[i] = b[i]
#
#     guess = np.zeros((nb_of_unknowns), dtype=int)  # creat an initial guess of the solution with all values set to zero
#     if verbose and solution is not None:
#         nb_correct = np.count_nonzero(solution == guess)
#         print("Number of correctly guessed unknowns: {:d}/{:d}"
#               .format(nb_correct, len(solution)))
#     if nb_of_inequalities == 0:
#         return guess
#     nb_of_values = 2 * eta + 1
#     x = np.arange(-eta, eta + 1, dtype=np.int8)
#     x_pmf = uniform.cdf(x + 1, -eta, 2 * eta + 1) - uniform.cdf(x, -eta, 2 * eta + 1)
#     print(x_pmf)
#     x_pmf = np.repeat(x_pmf.reshape(1, -1), nb_of_unknowns,
#                       axis=0)  # this line repeats the x_pmf array multiple times to creat a 2D array
#     # print(x_pmf[1])
#     a = a.astype(np.int16)  # this change the datatype of the matrix a to int16
#     a_squared = np.square(a)  # this squares each element of a
#
#     prob_geq_zero = np.zeros((nb_of_inequalities), dtype=float)
#     p_failure_is_observed = np.count_nonzero(is_geq_zero) / nb_of_inequalities
#     print("p_failure_is_observed", p_failure_is_observed)
#     mean = np.matmul(x_pmf, x)
#     variance = np.matmul(x_pmf, np.square(x)) - np.square(mean)
#     mean = np.matmul(a, mean)  # adjust the mean and variance using matrix multiplication
#     variance = np.matmul(a_squared, variance)
#     zscore = np.divide(mean + 0.5 + b, np.sqrt(variance))
#     p_failure_is_reality = norm.cdf(zscore)  # central limit theorem
#     p_failure_is_reality = np.mean(p_failure_is_reality)
#     print("p_failure_is_reality", p_failure_is_reality)
#     p_inequality_is_correct = min(
#         p_failure_is_reality / p_failure_is_observed, 1.0)
#     print("p_inequality_is_correct", p_inequality_is_correct)
#     prob_geq_zero[is_geq_zero] = p_inequality_is_correct
#     # print("prob_geq_zero",prob_geq_zero)
#     fitness = np.zeros((max_nb_of_iterations), dtype=float)
#     fitness_max = np.sum(np.maximum(prob_geq_zero, 1 - prob_geq_zero))  # 正确>0的个数
#     # print(guess[1])
#     count = [0] * max_nb_of_iterations
#     for z in range(max_nb_of_iterations):
#         if verbose:
#             print("Iteration " + str(z))
#             time_start = time.time()
#         mean = np.matmul(x_pmf, x)  # 计算当前分布下，所有未知数的期望值
#         variance = np.matmul(x_pmf, np.square(x)) - np.square(mean)
#         mean = np.multiply(a, np.repeat(mean[np.newaxis, :],
#                                         nb_of_inequalities, axis=0))
#         variance = np.multiply(
#             a_squared,
#             np.repeat(variance[np.newaxis, :], nb_of_inequalities, axis=0))
#         mean = mean.sum(axis=1).reshape(-1, 1).repeat(nb_of_unknowns, axis=1) \
#                - mean
#         mean -= b[:, np.newaxis]
#         variance = variance.sum(axis=1).reshape(-1, 1).repeat(nb_of_unknowns,
#                                                               axis=1) - variance
#         variance = np.clip(variance, 1, None)
#         psuccess = np.zeros((nb_of_values, nb_of_inequalities,
#                              nb_of_unknowns), dtype=float)
#         for j in range(nb_of_values):
#             zscore = np.divide(a * x[j] + mean + 0.5, np.sqrt(variance))
#             psuccess[j, :, :] = norm.cdf(zscore)  # central limit theorem
#         psuccess = np.transpose(psuccess, axes=[2, 0, 1])
#         psuccess = \
#             np.multiply(psuccess, prob_geq_zero[np.newaxis, np.newaxis, :]) + \
#             np.multiply(1 - psuccess, 1 - prob_geq_zero[np.newaxis, np.newaxis, :])
#         # psuccess = np.clip(psuccess, 10e-15, None)
#         psuccess = np.sum(np.log(psuccess), axis=2)
#         row_means = psuccess.max(axis=1)
#         psuccess -= row_means[:, np.newaxis]
#         psuccess = np.exp(psuccess)
#
#         x_pmf = np.multiply(psuccess, x_pmf)
#         row_sums = x_pmf.sum(axis=1)
#         x_pmf /= row_sums[:, np.newaxis]
#         print(x_pmf[:3])
#         max_pro = np.max(x_pmf, axis=1)
#         # print(max_pro)S
#         # 对数组进行降序排序
#         sorted_indices = np.argsort(max_pro)[::-1]
#         # 选择前512个最大元素的索引
#         index_pro = sorted_indices[:512]
#         # print(index_pro)
#         guess = x[np.argmax(x_pmf, axis=1)]
#         fit = (np.matmul(a, guess) + b >= 0).astype(float)
#         fit = np.dot(fit, prob_geq_zero) + np.dot(1 - fit, 1 - prob_geq_zero)
#         fitness[z] = fit / fitness_max
#         if z == max_nb_of_iterations - 1:
#             print(np.array(guess))
#         if verbose:
#             time_end = time.time()
#             print("Elapsed time: {:.1f} seconds".format(time_end - time_start))
#             print("Fitness {:.2f}%".format(fitness[z] * 100))
#             if solution is not None:
#                 nb_correct = np.count_nonzero(solution == guess)
#                 count[z] = nb_correct
#                 print("Number of correctly guessed unknowns: {:d}/{:d}"
#                       .format(nb_correct, len(solution)))
#         # 使用index_pro
#         selected_guess = guess[index_pro]
#         selected_solution = solution[index_pro]
#         matches = selected_guess == selected_solution
#         num_of_matches = np.count_nonzero(matches)
#         print("Number of selected coeffs matches:{:d}/512"
#               .format(num_of_matches))
#         if (z > 1) and count[z - 1] >= count[z] + 1:
#             print(np.array(guess))
#             break
#
#     return guess
#
#
# # Eliminate the failure rate
# def solve_inequalities_SMY2(kem, a, b, is_geq_zero,
#                             max_nb_of_iterations=10,
#                             verbose=True,
#                             solution=None):  # analyze convergence rate with a known solution
#     if verbose:
#         print("Solving inequalities...")
#     eta = kem.pke.ETA1
#     [nb_of_inequalities, nb_of_unknowns] = a.shape
#     guess = np.zeros((nb_of_unknowns), dtype=int)  # creat an initial guess of the solution with all values set to zero
#     if verbose and solution is not None:
#         nb_correct = np.count_nonzero(solution == guess)
#         print("Number of correctly guessed unknowns: {:d}/{:d}"
#               .format(nb_correct, len(solution)))
#     if nb_of_inequalities == 0:
#         return guess
#     nb_of_values = 2 * eta + 1
#     x = np.arange(-eta, eta + 1, dtype=np.int8)
#     x_pmf = binom.pmf(x + eta, 2 * eta, 0.5)
#     print(x_pmf)
#     x_pmf = np.repeat(x_pmf.reshape(1, -1), nb_of_unknowns,
#                       axis=0)  # this line repeats the x_pmf array multiple times to creat a 2D array
#     # print(x_pmf[1])
#     a = a.astype(np.int16)  # this change the datatype of the matrix a to int16
#     a_squared = np.square(a)  # this squares each element of a
#
#     count = [0] * max_nb_of_iterations
#     for z in range(max_nb_of_iterations):
#         if verbose:
#             print("Iteration " + str(z))
#             time_start = time.time()
#         mean = np.matmul(x_pmf, x)  # 计算当前分布下，所有未知数的期望值
#         variance = np.matmul(x_pmf, np.square(x)) - np.square(mean)
#         mean = np.multiply(a, np.repeat(mean[np.newaxis, :],
#                                         nb_of_inequalities, axis=0))
#         variance = np.multiply(
#             a_squared,
#             np.repeat(variance[np.newaxis, :], nb_of_inequalities, axis=0))
#         mean = mean.sum(axis=1).reshape(-1, 1).repeat(nb_of_unknowns, axis=1) - mean
#         # print("mean",mean[:3])
#         mean += b[:, np.newaxis]
#         # print("mean", mean[:3])
#         variance = variance.sum(axis=1).reshape(-1, 1).repeat(nb_of_unknowns, axis=1) - variance
#         variance = np.clip(variance, 1, None)
#         psuccess = np.zeros((nb_of_values, nb_of_inequalities,
#                              nb_of_unknowns), dtype=float)
#         for j in range(nb_of_values):
#             # print("a * x[j]",(a * x[j])[0])
#             zscore = np.divide(a * x[j] + mean + 0.5, np.sqrt(variance))
#             # print("norm.cdf(zscore)", norm.cdf(zscore)[0])
#             psuccess[j, :, :] = norm.cdf(zscore)  # central limit theorem
#         psuccess = np.transpose(psuccess, axes=[2, 0, 1])
#         psuccess = \
#             np.multiply(psuccess, is_geq_zero) + \
#             np.multiply(1 - psuccess, 1 - is_geq_zero)
#         # print("psuccess", psuccess[0])
#         psuccess = np.clip(psuccess, 10e-5, None)
#         psuccess = np.sum(np.log(psuccess), axis=2)
#         row_means = psuccess.max(axis=1)
#         psuccess -= row_means[:, np.newaxis]
#         psuccess = np.exp(psuccess)
#
#         x_pmf = np.multiply(psuccess, x_pmf)
#         row_sums = x_pmf.sum(axis=1)
#         x_pmf /= row_sums[:, np.newaxis]
#         # print(x_pmf[:3])
#         max_pro = np.max(x_pmf, axis=1)
#         # print(max_pro)
#         # 对数组进行降序排序
#         sorted_indices = np.argsort(max_pro)[::-1]
#         # 选择前512个最大元素的索引
#         index_pro = sorted_indices[:512]
#         # print(index_pro)
#         guess = x[np.argmax(x_pmf, axis=1)]
#
#         if z == max_nb_of_iterations - 1:
#             print(np.array(guess))
#         if verbose:
#             time_end = time.time()
#             print("Elapsed time: {:.1f} seconds".format(time_end - time_start))
#             if solution is not None:
#                 nb_correct = np.count_nonzero(solution == guess)
#                 count[z] = nb_correct
#                 print("Number of correctly guessed unknowns: {:d}/{:d}"
#                       .format(nb_correct, len(solution)))
#
#         # 使用index_pro
#         selected_guess = guess[index_pro]
#         selected_solution = solution[index_pro]
#         matches = selected_guess == selected_solution
#         num_of_matches = np.count_nonzero(matches)
#         print("Number of selected coeffs matches:{:d}/512"
#               .format(num_of_matches))
#
#         if (z > 1) and count[z - 1] >= count[z] + 1:
#             # print(np.array(guess))
#             break
#
#     return guess
#
#
# # block BP with block size 2
# def solve_inequalities_block2(kem, a, b, is_geq_zero,
#                               max_nb_of_iterations=20,
#                               verbose=True,
#                               solution=None, block_size=2):  # analyze convergence rate with a known solution
#     if verbose:
#         print("Solving inequalities...")
#     eta = kem.pke.ETA1
#     [nb_of_inequalities, nb_of_unknowns] = a.shape
#     guess = np.zeros(nb_of_unknowns, dtype=int)  # creat an initial guess of the solution with all values set to zero
#     if verbose and solution is not None:
#         nb_correct = np.count_nonzero(solution == guess)
#         print("Number of correctly guessed unknowns: {:d}/{:d}"
#               .format(nb_correct, len(solution)))
#     if nb_of_inequalities == 0:
#         return guess
#
#     nb_of_values = 2 * eta + 1
#     x = np.arange(-eta, eta + 1, dtype=np.int8)
#     x_pmf = binom.pmf(x + eta, 2 * eta, 0.5)
#     print(x_pmf)
#
#     nb_of_block = int(nb_of_unknowns / block_size)
#     block_value = [(i, j) for i in x for j in x]
#     block_pmf = np.zeros(len(block_value))
#     for index, (i, j) in enumerate(block_value):
#         block_pmf[index] = x_pmf[i + eta] * x_pmf[j + eta]
#     # print("block_value", block_value)
#     # print("block_pmf",block_pmf)
#     block_pmf = np.repeat(block_pmf.reshape(1, -1), nb_of_block, axis=0)
#     print("block_pmf.shape", block_pmf.shape)
#
#     x_pmf = np.repeat(x_pmf.reshape(1, -1), nb_of_unknowns, axis=0)
#     print("x_pmf.shape", x_pmf.shape)
#     a = a.astype(np.int16)  # this change the datatype of the matrix a to int16
#     a_squared = np.square(a)  # this squares each element of a
#
#     count = [0] * max_nb_of_iterations
#     for z in range(max_nb_of_iterations):
#         if verbose:
#             print("Iteration " + str(z))
#             time_start = time.time()
#
#         # mean = []
#         # variance = []
#         # for _ in range(nb_of_block):
#         #     mean1 = 0
#         #     mean2 = 0
#         #     variance1 = 0
#         #     variance2 = 0
#         #     for (value1, value2), pmf in zip(block_value, block_pmf):
#         #         mean1 += value1 * pmf
#         #         mean2 += value2 * pmf
#         #     for (value1, value2), pmf in zip(block_value, block_pmf):
#         #         variance1 += (value1-mean1)**2 * pmf
#         #         variance2 += (value2-mean2)**2 * pmf
#         #     mean.append(mean1)
#         #     mean.append(mean2)
#         #     variance.append(variance1)
#         #     variance.append(variance2)
#         #
#         # mean = np.array(mean)
#         # mean = np.multiply(a, np.repeat(mean[np.newaxis, :], nb_of_inequalities, axis=0))
#         # variance = np.multiply(
#         #     a_squared,
#         #     np.repeat(variance[np.newaxis, :], nb_of_inequalities, axis=0))
#         mean = np.matmul(x_pmf, x)  # 计算当前分布下，所有未知数的期望值
#         # print("mean.shape", mean.shape)
#         variance = np.matmul(x_pmf, np.square(x)) - np.square(mean)
#         mean = np.multiply(a, np.repeat(mean[np.newaxis, :], nb_of_inequalities, axis=0))
#         # print("mean.shape", mean.shape)
#         variance = np.multiply(a_squared, np.repeat(variance[np.newaxis, :], nb_of_inequalities, axis=0))
#         # print("variance.shape", variance.shape)
#
#         mean_block = np.zeros((nb_of_inequalities, nb_of_block))
#         mean_sum = mean.sum(axis=1).reshape(-1, 1)
#         # print("mean_sum.shape",mean_sum.shape)
#         for i in range(nb_of_block):
#             mean_block[:, i] = mean_sum[:, 0] - mean[:, 2 * i] - mean[:, 2 * i + 1]
#         mean_block += b[:, np.newaxis]
#         # print("mean_block.shape", mean_block.shape)
#
#         variance_block = np.zeros((nb_of_inequalities, nb_of_block))
#         variance_sum = variance.sum(axis=1).reshape(-1, 1)
#         for i in range(nb_of_block):
#             variance_block[:, i] = variance_sum[:, 0] - variance[:, 2 * i] - variance[:, 2 * i + 1]
#         # print("variance_block.shape", variance_block.shape)
#         # print(variance_block)
#
#         variance_block = np.clip(variance_block, 1, None)
#         psuccess = np.zeros((nb_of_values ** 2, nb_of_inequalities,
#                              nb_of_block), dtype=float)
#         for i in range(nb_of_values ** 2):
#             index_i = int(i / nb_of_values)
#             index_j = int(i % nb_of_values)
#             zscore = np.zeros((nb_of_inequalities, nb_of_block))
#             for j in range(nb_of_block):
#                 zscore[:, j] = np.divide(
#                     a[:, 2 * j] * x[index_i] + a[:, 2 * j + 1] * x[index_j] + mean_block[:, j] + 0.5,
#                     np.sqrt(variance_block[:, j]))
#             psuccess[i, :, :] = norm.cdf(zscore)  # central limit theorem
#
#         psuccess = np.transpose(psuccess, axes=[2, 0, 1])
#         psuccess = np.multiply(psuccess, is_geq_zero) + np.multiply(1 - psuccess, 1 - is_geq_zero)
#         # print("psuccess", psuccess[0])
#         psuccess = np.clip(psuccess, 10e-5, None)
#         psuccess = np.sum(np.log(psuccess), axis=2)
#         row_means = psuccess.max(axis=1)  # 数值稳定化处理
#         psuccess -= row_means[:, np.newaxis]  # 数值稳定化处理
#         psuccess = np.exp(psuccess)
#         # print("psuccess.shape", psuccess.shape)
#
#         # print("block_pmf.shape", block_pmf.shape)
#         block_pmf = np.multiply(psuccess, block_pmf)
#         # print("block_pmf.shape", block_pmf.shape)
#         row_sums = block_pmf.sum(axis=1)
#         block_pmf /= row_sums[:, np.newaxis]  # 归一化处理
#         # print("block_pmf.shape", block_pmf.shape)
#
#         # 更新x_pmf的分布
#         for i in range(nb_of_block):
#             for j in range(nb_of_values):
#                 x_pmf[2 * i][j] = block_pmf[i][j * nb_of_values:(j + 1) * nb_of_values].sum()
#                 indices = np.arange(j, 49, 7)
#                 x_pmf[2 * i + 1][j] = block_pmf[i][indices].sum()
#         # print("x_pmf[0]", x_pmf[0])
#         # print("x_pmf[1]", x_pmf[1])
#         # print("x_pmf[2]", x_pmf[2])
#         # print("x_pmf[3]", x_pmf[3])
#
#         print("block_pmf[0]", block_pmf[0])
#         print("block_pmf[1]", block_pmf[1])
#         print("block_pmf[2]", block_pmf[2])
#         print("block_pmf[3]", block_pmf[3])
#
#         # print(x_pmf[:3])
#         max_pro = np.max(block_pmf, axis=1)
#         # print(max_pro)
#         # 对数组进行降序排序
#         sorted_indices = np.argsort(max_pro)[::-1]
#         # 选择前512个最大元素的索引
#         block_index_pro = sorted_indices[:256]
#         index_pro = np.zeros(512, dtype=int)
#         for i in range(256):
#             index_pro[2 * i] = 2 * block_index_pro[i]
#             index_pro[2 * i + 1] = 2 * block_index_pro[i] + 1
#         # print(index_pro)
#
#         guess = np.zeros(nb_of_unknowns, dtype=int)
#         res = np.argmax(block_pmf, axis=1)
#         for i in range(nb_of_block):
#             index_i = int(res[i] / nb_of_values)
#             index_j = int(res[i] % nb_of_values)
#             guess[2 * i] = x[index_i]
#             guess[2 * i + 1] = x[index_j]
#         print("guess", guess)
#
#         if z == max_nb_of_iterations - 1:
#             print(np.array(guess))
#         if verbose:
#             time_end = time.time()
#             print("Elapsed time: {:.1f} seconds".format(time_end - time_start))
#             if solution is not None:
#                 nb_correct = np.count_nonzero(solution == guess)
#                 count[z] = nb_correct
#                 print("Number of correctly guessed unknowns: {:d}/{:d}"
#                       .format(nb_correct, len(solution)))
#
#         # 使用index_pro
#         selected_guess = guess[index_pro]
#         selected_solution = solution[index_pro]
#         matches = selected_guess == selected_solution
#         num_of_matches = np.count_nonzero(matches)
#         print("Number of selected coeffs matches:{:d}/512"
#               .format(num_of_matches))
#
#         # if (z > 1) and count[z-1] >= count[z] + 1:
#         #     # print(np.array(guess))
#         #     break
#
#     return guess
#
#
# # Eliminate the failure rate, and the secret from uniform distribution
# def solve_inequalities_SMY3(eta, a, b, is_geq_zero,
#                             max_nb_of_iterations=2,
#                             verbose=True,
#                             solution=None):  # analyze convergence rate with a known solution
#     if verbose:
#         print("Solving inequalities...")
#     eta = eta
#     [nb_of_inequalities, nb_of_unknowns] = a.shape
#     guess = np.zeros((nb_of_unknowns), dtype=int)  # creat an initial guess of the solution with all values set to zero
#     if verbose and solution is not None:
#         nb_correct = np.count_nonzero(solution == guess)
#         print("Number of correctly guessed unknowns: {:d}/{:d}"
#               .format(nb_correct, len(solution)))
#     if nb_of_inequalities == 0:
#         return guess
#     nb_of_values = 2 * eta + 1
#     x = np.arange(-eta, eta + 1, dtype=np.int8)
#     x_pmf = uniform.cdf(x + 1, -eta, 2 * eta + 1) - uniform.cdf(x, -eta, 2 * eta + 1)
#     # print(x_pmf)
#     x_pmf = np.repeat(x_pmf.reshape(1, -1), nb_of_unknowns,
#                       axis=0)  # this line repeats the x_pmf array multiple times to creat a 2D array
#     # print(x_pmf[1])
#     a = a.astype(np.int16)  # this change the datatype of the matrix a to int16
#     a_squared = np.square(a)  # this squares each element of a
#
#     for z in range(max_nb_of_iterations):
#         if verbose:
#             print("Iteration " + str(z))
#             time_start = time.time()
#         mean = np.matmul(x_pmf, x)  # 计算当前分布下，所有未知数的期望值
#         variance = np.matmul(x_pmf, np.square(x)) - np.square(mean)
#         mean = np.multiply(a, np.repeat(mean[np.newaxis, :],
#                                         nb_of_inequalities, axis=0))
#         variance = np.multiply(
#             a_squared,
#             np.repeat(variance[np.newaxis, :], nb_of_inequalities, axis=0))
#
#         mean = mean.sum(axis=1).reshape(-1, 1).repeat(nb_of_unknowns, axis=1) \
#                - mean
#         # print("mean",mean[:3])
#         mean += b[:, np.newaxis]
#         # print("mean", mean[:3])
#
#         variance = variance.sum(axis=1).reshape(-1, 1).repeat(nb_of_unknowns,
#                                                               axis=1) - variance
#         variance = np.clip(variance, 1, None)
#         psuccess = np.zeros((nb_of_values, nb_of_inequalities,
#                              nb_of_unknowns), dtype=float)
#         for j in range(nb_of_values):
#             zscore = np.divide(a * x[j] + mean + 0.5, np.sqrt(variance))
#             # print("norm.cdf(zscore)", norm.cdf(zscore)[:3])
#             # psuccess[j, :, :] = norm.cdf(zscore) / norm.cdf(np.divide(mean + 0.5, np.sqrt(variance))) # central limit theorem
#             psuccess[j, :, :] = norm.cdf(zscore)
#         psuccess = np.transpose(psuccess, axes=[2, 0, 1])
#         psuccess = \
#             np.multiply(psuccess, is_geq_zero) + \
#             np.multiply(1 - psuccess, 1 - is_geq_zero)
#         # print("psuccess", psuccess[0])
#         # psuccess = np.clip(psuccess, 10e-5, None)
#         psuccess = np.sum(np.log(psuccess), axis=2)
#         row_means = psuccess.max(axis=1)
#         psuccess -= row_means[:, np.newaxis]
#         psuccess = np.exp(psuccess)
#
#         x_pmf = np.multiply(psuccess, x_pmf)
#         row_sums = x_pmf.sum(axis=1)
#         x_pmf /= row_sums[:, np.newaxis]
#         # print(x_pmf[:3])
#         max_pro = np.max(x_pmf, axis=1)
#         # print(max_pro)
#         # 对数组进行降序排序
#         sorted_indices = np.argsort(max_pro)[::-1]
#         # 选择前512个最大元素的索引
#         index_pro = sorted_indices[:512]
#         # print(index_pro)
#         guess = x[np.argmax(x_pmf, axis=1)]
#
#         # if z == max_nb_of_iterations - 1:
#         #     print(np.array(guess))
#         if verbose:
#             time_end = time.time()
#             print("Elapsed time: {:.1f} seconds".format(time_end - time_start))
#             if solution is not None:
#                 nb_correct = np.count_nonzero(solution == guess)
#                 print("Del22, Number of correctly guessed unknowns: {:d}/{:d}"
#                       .format(nb_correct, len(solution)))
#
#         # 使用index_pro
#         selected_guess = guess[index_pro]
#         selected_solution = solution[index_pro]
#         matches = selected_guess == selected_solution
#         num_of_matches = np.count_nonzero(matches)
#         print("Number of selected coeffs matches:{:d}/512"
#               .format(num_of_matches))
#
#     return guess, nb_correct
#
#
# # Eliminate the failure rate, and the secret from uniform distribution
# def solve_inequalities_SMY4(eta, a, b, is_geq_zero, solution):
#     # print("Solving inequalities...")
#
#     [nb_of_inequalities, nb_of_unknowns] = a.shape
#
#     # as>b
#     for i in range(0, nb_of_inequalities):
#         if is_geq_zero[i]:
#             a[i] = np.array(a[i])
#             b[i] = b[i]
#         else:
#             a[i] = np.array(-a[i])
#             b[i] = -b[i]
#
#     guess = np.zeros(nb_of_unknowns, dtype=int)  # creat an initial guess of the solution with all values set to zero
#
#     if nb_of_inequalities == 0:
#         return guess
#     nb_of_values = 2 * eta + 1
#     x = np.arange(-eta, eta + 1, dtype=np.int8)
#     # x_pmf = binom.pmf(x + eta, 2 * eta, 0.5)
#     x_pmf = uniform.cdf(x + 1, -eta, 2 * eta + 1) - uniform.cdf(x, -eta, 2 * eta + 1)
#     # print("x_pmf", x_pmf)
#     x_pmf = np.repeat(x_pmf.reshape(1, -1), nb_of_unknowns,
#                       axis=0)  # this line repeats the x_pmf array multiple times to creat a 2D array
#     a = a.astype(np.int16)  # this change the datatype of the matrix a to int16
#     a_squared = np.square(a)  # this squares each element of a
#
#     mean = np.matmul(x_pmf, x)  # 计算当前分布下，所有未知数的期望值
#     variance = np.matmul(x_pmf, np.square(x)) - np.square(mean)  # 方差计算公式
#     mean = np.multiply(a, np.repeat(mean[np.newaxis, :],
#                                     nb_of_inequalities, axis=0))
#     variance = np.multiply(
#         a_squared,
#         np.repeat(variance[np.newaxis, :], nb_of_inequalities, axis=0))
#     # mean_all = mean.sum(axis=1).reshape(-1, 1).repeat(nb_of_unknowns, axis=1)
#     mean = mean.sum(axis=1).reshape(-1, 1).repeat(nb_of_unknowns, axis=1) - mean
#     # print("mean", mean)
#     # mean -= b[:, np.newaxis]
#     # mean_all += b[:, np.newaxis]
#     mean += 800  # 实验表明，750， 800效果最佳
#     # variance_all = variance.sum(axis=1).reshape(-1, 1).repeat(nb_of_unknowns, axis=1)
#     variance = variance.sum(axis=1).reshape(-1, 1).repeat(nb_of_unknowns, axis=1) - variance
#     # print("variance", variance)
#     variance = np.clip(variance, 1, None)
#     psuccess = np.zeros((nb_of_values, nb_of_inequalities,
#                          nb_of_unknowns), dtype=float)
#     for j in range(nb_of_values):
#         zscore = np.divide(a * x[j] + mean + 0.5, np.sqrt(variance))
#         # zscore = np.divide(a * x[j] + mean + 6*np.sqrt(variance), np.sqrt(variance))
#         # print("norm.cdf(zscore)", norm.cdf(np.divide(100, np.sqrt(14500))))
#         # psuccess[j, :, :] = norm.cdf(zscore) / norm.cdf(np.divide(mean + 0.5, np.sqrt(variance))) # central limit theorem
#         psuccess[j, :, :] = norm.cdf(zscore)  # / norm.cdf(np.divide(mean_all, np.sqrt(variance_all)))
#         # print("psuccess[j, 0, 0]", psuccess[j, 0, 0])
#     psuccess = np.transpose(psuccess, axes=[2, 0, 1])
#     psuccess = np.sum(np.log(psuccess), axis=2)
#     # print("psuccess", psuccess)
#     psuccess = np.exp(psuccess)
#     # psuccess = np.sum(psuccess)
#
#     x_pmf = np.multiply(psuccess, x_pmf)
#     # print("x_pmf", x_pmf)
#     row_sums = x_pmf.sum(axis=1)
#     x_pmf /= row_sums[:, np.newaxis]
#
#     # for i in range(3):
#     #     print("x_pmf%d" %i)
#     #     for element in x_pmf[i]:
#     #         print(f"{element:.16f}")
#     guess_ave = np.matmul(x_pmf, x)
#     # print("guess_ave", guess_ave)
#
#     max_pro = np.max(x_pmf, axis=1)
#     # print("max_pro", max_pro)
#     # print("max_pro_0,max_pro_1,max_pro_2,max_pro_3)", max_pro[0],max_pro[1], max_pro[2],max_pro[3])
#     # 对数组进行降序排序
#     sorted_indices = np.argsort(max_pro)[::-1]
#     # print("sorted_indices", sorted_indices)
#     # 选择前512个最大元素的索引
#     index_pro = sorted_indices[:512]
#     guess = x[np.argmax(x_pmf, axis=1)]
#     # print("guess ", guess)
#
#     if solution is not None:
#         nb_correct = np.count_nonzero(solution == guess)
#         # print("Number of correctly guessed unknowns: {:d}/{:d}".format(nb_correct, len(solution)))
#
#     # 使用index_pro
#     selected_guess = guess[index_pro]
#     selected_solution = solution[index_pro]
#     matches = selected_guess == selected_solution
#     num_of_matches = np.count_nonzero(matches)
#     # print("Number of selected coeffs matches:{:d}/512".format(num_of_matches))
#
#     return guess, nb_correct
#
#
# def solve_inequalities_DGJ19(eta, a, solution):
#     # print("Solving inequalities...")
#     qt = 3329 / 4
#     [nb_of_inequalities, nb_of_unknowns] = a.shape
#     # print("the nb_of_inequalities", nb_of_inequalities)
#     guess = np.zeros(nb_of_unknowns, dtype=int)  # creat an initial guess of the solution with all values set to zero
#
#     if nb_of_inequalities == 0:
#         return guess
#     nb_of_values = 2 * eta + 1
#     x = np.arange(-eta, eta + 1, dtype=np.int8)
#     x_pmf = binom.pmf(x + eta, 2 * eta, 0.5)
#     # print("x_pmf", x_pmf)
#     x_pmf = np.repeat(x_pmf.reshape(1, -1), nb_of_unknowns,
#                       axis=0)  # this line repeats the x_pmf array multiple times to creat a 2D array
#     a = a.astype(np.int16)  # this change the datatype of the matrix a to int16
#     a_squared = np.square(a)  # this squares each element of a
#
#     mean = np.matmul(x_pmf, x)  # 计算当前分布下，所有未知数的期望值
#     variance = np.matmul(x_pmf, np.square(x)) - np.square(mean)  # 方差计算公式
#     mean = np.multiply(a, np.repeat(mean[np.newaxis, :],
#                                     nb_of_inequalities, axis=0))
#     variance = np.multiply(
#         a_squared,
#         np.repeat(variance[np.newaxis, :], nb_of_inequalities, axis=0))
#     # mean_all = mean.sum(axis=1).reshape(-1, 1).repeat(nb_of_unknowns, axis=1)
#     mean = mean.sum(axis=1).reshape(-1, 1).repeat(nb_of_unknowns, axis=1) - mean
#     # print("mean", mean)
#     # mean -= b[:, np.newaxis]
#     # mean_all += b[:, np.newaxis]
#     # mean += qt # 实验表明，750， 800效果最佳
#     variance = variance.sum(axis=1).reshape(-1, 1).repeat(nb_of_unknowns, axis=1) - variance
#     # print("variance", variance)
#     variance = np.clip(variance, 1, None)
#     psuccess = np.zeros((nb_of_values, nb_of_inequalities,
#                          nb_of_unknowns), dtype=float)
#     for j in range(nb_of_values):
#         zscore = np.divide(a * x[j] + mean + 0.5, np.sqrt(variance))
#         # zscore = np.divide(a * x[j] + mean + 6*np.sqrt(variance), np.sqrt(variance))
#         # print("norm.cdf(zscore)", norm.cdf(np.divide(100, np.sqrt(14500))))
#         # psuccess[j, :, :] = norm.cdf(zscore) / norm.cdf(np.divide(mean + 0.5, np.sqrt(variance))) # central limit theorem
#         psuccess[j, :, :] = norm.cdf(zscore)  # / norm.cdf(np.divide(mean_all, np.sqrt(variance_all)))
#         # print("psuccess[j, 0, 0]", psuccess[j, 0, 0])
#     psuccess = np.transpose(psuccess, axes=[2, 0, 1])
#     psuccess = np.sum(np.log(psuccess), axis=2)
#     # print("psuccess", psuccess)
#     psuccess = np.exp(psuccess)
#     # psuccess = np.sum(psuccess)
#
#     x_pmf = np.multiply(psuccess, x_pmf)
#     # print("x_pmf", x_pmf)
#     row_sums = x_pmf.sum(axis=1)
#     x_pmf /= row_sums[:, np.newaxis]
#
#     # for i in range(3):
#     #     print("x_pmf%d" %i)
#     #     for element in x_pmf[i]:
#     #         print(f"{element:.16f}")
#     # guess_ave = np.matmul(x_pmf, x)
#     # print("guess_ave", guess_ave)
#
#     # max_pro = np.max(x_pmf, axis=1)
#     # print("max_pro", max_pro)
#     # print("max_pro_0,max_pro_1,max_pro_2,max_pro_3)", max_pro[0],max_pro[1], max_pro[2],max_pro[3])
#     # 对数组进行降序排序
#     # sorted_indices = np.argsort(max_pro)[::-1]
#     # print("sorted_indices", sorted_indices)
#     # 选择前512个最大元素的索引
#     # index_pro = sorted_indices[:512]
#     guess = x[np.argmax(x_pmf, axis=1)]
#     # print("guess ", guess)
#
#     if solution is not None:
#         nb_correct = np.count_nonzero(solution == guess)
#         print("Number of correctly guessed unknowns: {:d}/{:d}".format(nb_correct, len(solution)))
#
#     # 使用index_pro
#     # selected_guess = guess[index_pro]
#     # selected_solution = solution[index_pro]
#     # matches = selected_guess == selected_solution
#     # num_of_matches = np.count_nonzero(matches)
#     # print("Number of selected coeffs matches:{:d}/512".format(num_of_matches))
#
#     return guess, nb_correct
#
#
# # estimate the secret key using these dfs on average
# def solve_inequalities_ARV20(a, solution=None):  # analyze convergence rate with a known solution
#     # print("estimate the secret key using these dfs on average in DRV20")
#     qt = 3329 / 4
#     [nb_of_inequalities, nb_of_unknowns] = a.shape
#     # print("nb_of_inequalities",nb_of_inequalities, "nb_of_unknowns", nb_of_unknowns)
#     guess = np.zeros(nb_of_unknowns, dtype=int)
#     failure_sum = [0] * nb_of_unknowns
#     failure = []
#     for i in range(0, nb_of_inequalities):
#         df = np.array(a[i])
#         failure.append(df)
#         failure_sum += np.array(df)
#     E = [0] * nb_of_unknowns
#     for i in range(0, nb_of_inequalities):
#         E += np.array(failure[i] / np.linalg.norm(failure[i]))
#
#     sest_norm = nb_of_inequalities * qt / np.linalg.norm(failure_sum)
#     E = np.array(normalize(E)) * sest_norm
#     E_int = [0] * nb_of_unknowns
#     for i in range(0, nb_of_unknowns):  # round numbers of secret
#         E_int[i] = round(E[i])
#
#         if E_int[i] >= 3:
#             E_int[i] = 3
#         elif E_int[i] <= -3:
#             E_int[i] = -3
#     E_int = np.array(E_int)
#     # print("E_int", E_int)
#     # print("E_int", solution)
#     if solution is not None:
#         nb_correct = np.count_nonzero(solution == E_int)
#         # print("Number of correctly guessed unknowns by ARV20: {:d}/{:d}".format(nb_correct, len(solution)))
#     return E_int, nb_correct
#
#
# def solve_inequalities_ARV20_uniform(a, is_geq_zero, solution=None):  # analyze convergence rate with a known solution
#     print("estimate the secret key using these dfs on average in DRV20")
#     qt = 3329 / 4
#     # qt = 850
#     [nb_of_inequalities, nb_of_unknowns] = a.shape
#     print("nb_of_inequalities", nb_of_inequalities, "nb_of_unknowns", nb_of_unknowns)
#     guess = np.zeros(nb_of_unknowns, dtype=int)
#     failure_sum = [0] * nb_of_unknowns
#     failure = []
#     for i in range(0, nb_of_inequalities):
#         if is_geq_zero[i]:
#             df = np.array(a[i])
#             # print(df)
#         else:
#             df = np.array(-a[i])
#             # print(df)
#         # df = np.array(a[i])
#         failure.append(df)
#         failure_sum += np.array(df)
#     E = [0] * nb_of_unknowns
#     for i in range(0, nb_of_inequalities):
#         E += np.array(failure[i] / np.linalg.norm(failure[i]))
#
#     sest_norm = nb_of_inequalities * qt / np.linalg.norm(failure_sum)
#     E = np.array(normalize(E)) * sest_norm
#     E_int = [0] * nb_of_unknowns
#     for i in range(0, nb_of_unknowns):  # round numbers of secret
#         E_int[i] = round(E[i])
#         if E_int[i] >= 2:
#             E_int[i] = 2
#         elif E_int[i] <= -2:
#             E_int[i] = -2
#     E_int = np.array(E_int)
#     # print("E_int", E_int)
#     # print("E_int", solution)
#     if solution is not None:
#         nb_correct = np.count_nonzero(solution == E_int)
#         print("Number of correctly guessed unknowns by ARV20: {:d}/{:d}".format(nb_correct, len(solution)))
#     return E_int, nb_correct
#
#
# # estimate the secret key using these dfs on average firstly, and then change the initial distribution of the secret
# def solve_inequalities_ARV20_Del22(eta, a, b, is_geq_zero, max_nb_of_iterations=3, verbose=True,
#                                    solution=None):  # analyze convergence rate with a known solution
#     qt = 3329 / 4
#     [nb_of_inequalities, nb_of_unknowns] = a.shape
#     print("nb_of_inequalities", nb_of_inequalities, "nb_of_unknowns", nb_of_unknowns)
#     failure_sum = [0] * nb_of_unknowns
#     failure = []
#     for i in range(0, nb_of_inequalities):
#         if is_geq_zero[i]:
#             df = np.array(a[i])
#         else:
#             df = np.array(-a[i])
#         failure.append(df)
#         failure_sum += np.array(df)
#     E = [0] * nb_of_unknowns
#     for i in range(0, nb_of_inequalities):
#         E += np.array(failure[i] / np.linalg.norm(failure[i]))
#
#     sest_norm = nb_of_inequalities * qt / np.linalg.norm(failure_sum)
#     E = np.array(normalize(E)) * sest_norm
#     E_int = [0] * nb_of_unknowns
#     for i in range(0, nb_of_unknowns):  # round numbers of secret
#         E_int[i] = round(E[i])
#         if E_int[i] >= 3:
#             E_int[i] = 3
#         elif E_int[i] <= -3:
#             E_int[i] = -3
#     print("The secret estimitation by ARV20:", E_int)
#     if solution is not None:
#         nb_correct = np.count_nonzero(solution == E_int)
#         print("Number of correctly guessed unknowns by ARV20: {:d}/{:d}"
#               .format(nb_correct, len(solution)))
#
#     eta = eta
#     guess = np.zeros(nb_of_unknowns, dtype=int)  # creat an initial guess of the solution with all values set to zero
#     nb_of_values = 2 * eta + 1
#     x = np.arange(-eta, eta + 1, dtype=np.int8)
#     x_pmf = []
#     for i in range(0, nb_of_unknowns):
#         x_pmf.append(tar_uniform(eta, E_int[i]))
#     # print(x_pmf_0[:3])
#
#     a = a.astype(np.int16)  # this change the datatype of the matrix a to int16
#     a_squared = np.square(a)  # this squares each element of a
#
#     # x_pmf = uniform.cdf(x + 1, -eta, 2 * eta + 1) - uniform.cdf(x, -eta, 2 * eta + 1)
#     # x_pmf = np.repeat(x_pmf.reshape(1, -1), nb_of_unknowns,
#     #                   axis=0)  # this line repeats the x_pmf array multiple times to creat a 2D array
#
#     for z in range(max_nb_of_iterations):
#         if verbose:
#             print("Iteration " + str(z))
#             time_start = time.time()
#         mean = np.matmul(x_pmf, x)  # 计算当前分布下，所有未知数的期望值
#         variance = np.matmul(x_pmf, np.square(x)) - np.square(mean)
#         mean = np.multiply(a, np.repeat(mean[np.newaxis, :],
#                                         nb_of_inequalities, axis=0))
#         variance = np.multiply(
#             a_squared,
#             np.repeat(variance[np.newaxis, :], nb_of_inequalities, axis=0))
#         mean = mean.sum(axis=1).reshape(-1, 1).repeat(nb_of_unknowns, axis=1) \
#                - mean
#         # print("mean",mean[:3])
#         mean += b[:, np.newaxis]
#         # print("mean", mean[:3])
#         variance = variance.sum(axis=1).reshape(-1, 1).repeat(nb_of_unknowns,
#                                                               axis=1) - variance
#         variance = np.clip(variance, 1, None)
#         psuccess = np.zeros((nb_of_values, nb_of_inequalities,
#                              nb_of_unknowns), dtype=float)
#         for j in range(nb_of_values):
#             zscore = np.divide(a * x[j] + mean + 0.5, np.sqrt(variance))
#             # print("norm.cdf(zscore)", norm.cdf(zscore)[:3])
#             psuccess[j, :, :] = norm.cdf(zscore)  # central limit theorem
#         psuccess = np.transpose(psuccess, axes=[2, 0, 1])
#         psuccess = \
#             np.multiply(psuccess, is_geq_zero) + \
#             np.multiply(1 - psuccess, 1 - is_geq_zero)
#         # print("psuccess", psuccess[0])
#         psuccess = np.clip(psuccess, 10e-5, None)
#         psuccess = np.sum(np.log(psuccess), axis=2)
#         row_means = psuccess.max(axis=1)
#         psuccess -= row_means[:, np.newaxis]
#         psuccess = np.exp(psuccess)
#
#         x_pmf = np.multiply(psuccess, x_pmf)
#         row_sums = x_pmf.sum(axis=1)
#         x_pmf /= row_sums[:, np.newaxis]
#         # print(x_pmf[:3])
#         max_pro = np.max(x_pmf, axis=1)
#         # print(max_pro)
#         # 对数组进行降序排序
#         sorted_indices = np.argsort(max_pro)[::-1]
#         # 选择前512个最大元素的索引
#         index_pro = sorted_indices[:512]
#         # print(index_pro)
#         guess = x[np.argmax(x_pmf, axis=1)]
#
#         # if z == max_nb_of_iterations - 1:
#         #     print(np.array(guess))
#         if verbose:
#             time_end = time.time()
#             print("Elapsed time: {:.1f} seconds".format(time_end - time_start))
#             if solution is not None:
#                 nb_correct = np.count_nonzero(solution == guess)
#                 print("Number of correctly guessed unknowns: {:d}/{:d}"
#                       .format(nb_correct, len(solution)))
#
#         # 使用index_pro
#         selected_guess = guess[index_pro]
#         selected_solution = solution[index_pro]
#         matches = selected_guess == selected_solution
#         num_of_matches = np.count_nonzero(matches)
#         print("Number of selected coeffs matches:{:d}/512"
#               .format(num_of_matches))
#
#     return guess
#
#
# # Secret Estimation in [DGJ+19]
# # estimate the secret using the conditional probability
# def solve_inequalities_DGJ191(eta1, eta2, a, solution=None):
#     eta1 = eta1
#     eta2 = eta2
#     m, n = a.shape
#     qt = int(3329 / 4)
#
#     s_list = []  # the list of secret s, e, r
#     for i in range(0, 2 * eta1 + 1):
#         s_list.append(i - eta1)
#     s_dict = {}
#     for i in range(0, 2 * eta1 + 1):
#         s_dict[i - eta1] = binom.pmf(i, 2 * eta1, 0.5)
#     # print("s_dict: ", s_dict)
#
#     e1_list = []  # the list of secret s, e, r
#     for i in range(0, 2 * eta2 + 1):
#         e1_list.append(i - eta2)
#     e1_dict = {}
#     for i in range(0, 2 * eta2 + 1):
#         e1_dict[i - eta2] = binom.pmf(i, 2 * eta2, 0.5)
#     # print("e1_dict: ", e1_dict)
#
#     e1_u = law_convolution(e1_dict, e1_dict)
#     # print("e1_u: ", e1_u)
#
#     # make the Conditional probability distribution
#     er = law_product(s_dict, s_dict)
#     # print("er: ", er)
#
#     s_e1u = law_product(s_dict, e1_u)
#     # print("s_e1u: ", s_e1u)
#
#     tmp1 = iter_law_convolution(er, int(n / 2) - 1)  # !!tmp函数定义
#     tmp2 = iter_law_convolution(s_e1u, int(n / 2) - 1)
#     tmp = law_convolution(tmp1, tmp2)
#     # print(tmp)
#
#     sest_ave = [0] * n  # secret estimation using the original method of AGJ+19
#     sest_lar = [0] * n  # secret estimation using the largest value
#
#     tmp_e = law_convolution(tmp, s_e1u)
#     tmp_e = tail_probability(tmp_e)
#     # print("tmp_e", tmp_e)
#
#     for i in range(0, int(n / 2)):
#         fgivensci = []
#         for j in range(0, 2 * eta1 + 1):
#             pro_log = 0
#             for k in range(0, m):
#                 p = tmp_e[750 - s_list[j] * a[k][i]]
#                 # print("p", p)
#                 # p = np.clip(p, 10e-15, None)
#                 pro_log += np.log(p)
#             fgivensci.append(pro_log)
#             # print("pro_log", pro_log)
#
#         # fgivensci = np.array(fgivensci) / sum(fgivensci)
#         # print("fgivensci", fgivensci)
#         max_log_prob = np.max(fgivensci)
#         stable_fgivensci = fgivensci - max_log_prob
#         # 将对数概率转换回普通概率
#         fgivensci = np.exp(stable_fgivensci)
#
#         # 归一化概率
#         fgivensci = fgivensci / np.sum(fgivensci)
#         # print(fgivensci)
#
#         for j in range(0, 2 * eta1 + 1):
#             # compute the secret key using the average value
#             sest_ave[i] += (j - eta1) * fgivensci[j]
#         # compute the secret key using the value with the largest prob
#         sest_lar[i] = np.argmax(fgivensci) - eta1
#
#     tmp_s = law_convolution(tmp, er)
#     tmp_s = tail_probability(tmp_s)
#     for i in range(int(n / 2), n):
#         fgivensci = []
#         for j in range(0, 2 * eta1 + 1):
#             pro_log = 0
#             for k in range(0, m):
#                 p = tmp_s[750 - s_list[j] * a[k][i]]  # / tmp_s[qt + 1 - 50]
#                 # print("p", p)
#                 pro_log += np.log(p)
#             fgivensci.append(pro_log)
#             # fgivensci.append(np.exp(pro_log)),会导致数值下溢
#         # print("fgivensci", fgivensci)
#         max_log_prob = np.max(fgivensci)
#         stable_fgivensci = fgivensci - max_log_prob
#         # 将对数概率转换回普通概率
#         fgivensci = np.exp(stable_fgivensci)
#
#         # 归一化概率
#         fgivensci = fgivensci / np.sum(fgivensci)
#         # print(fgivensci)
#
#         for j in range(0, 2 * eta1 + 1):
#             # compute the secret key using the average value
#             sest_ave[i] += (j - eta1) * fgivensci[j]
#         # compute the secret key using the value with the largest prob
#         sest_lar[i] = np.argmax(fgivensci) - eta1
#
#     E_int_ave = [0] * n
#     for i in range(0, n):  # round numbers of secret
#         E_int_ave[i] = round(sest_ave[i])
#         if E_int_ave[i] > eta1:
#             E_int_ave[i] = eta1
#         elif E_int_ave[i] < -eta1:
#             E_int_ave[i] = -eta1
#     # print("sest_ave", E_int_ave)
#     # print("sest_lar", sest_lar)
#
#     matnum_ave = 0
#     matnum_lar = 0
#     for i in range(0, n):  # round numbers of secret
#         if E_int_ave[i] == solution[i]:
#             matnum_ave += 1
#         # if sest_lar[i] == solution[i]:
#         #     matnum_lar += 1
#
#     # print('DGJ+19, ciphertexts %d, experimental_matchnum using average value %d,'
#     #       ' experimental_matchnum using largest prob value %d' % (m, matnum_ave, matnum_lar))
#     print('DGJ+19, ciphertexts %d, experimental_matchnum using average value %d,' % (m, matnum_ave))
#
#     return sest_ave, matnum_ave
#
#
# def solve_inequalities_DGJ192(eta1, eta2, a, solution=None):
#     eta1 = eta1
#     eta2 = eta2
#     m, n = a.shape
#     qt = int(3329 / 4)
#
#     s_list = []  # the list of secret s, e, r
#     for i in range(0, 2 * eta1 + 1):
#         s_list.append(i - eta1)
#     s_dict = {}
#     for i in range(0, 2 * eta1 + 1):
#         s_dict[i - eta1] = binom.pmf(i, 2 * eta1, 0.5)
#     # print("s_dict: ", s_dict)
#
#     e1_list = []  # the list of secret s, e, r
#     for i in range(0, 2 * eta2 + 1):
#         e1_list.append(i - eta2)
#     e1_dict = {}
#     for i in range(0, 2 * eta2 + 1):
#         e1_dict[i - eta2] = binom.pmf(i, 2 * eta2, 0.5)
#     # print("e1_dict: ", e1_dict)
#
#     # e1_u = law_convolution(e1_dict, e1_dict)
#     # print("e1_u: ", e1_u)
#
#     # make the Conditional probability distribution
#     er = law_product(s_dict, s_dict)
#     # print("er: ", er)
#
#     s_e1u = law_product(s_dict, e1_dict)
#     # print("s_e1u: ", s_e1u)
#
#     tmp1 = iter_law_convolution(er, int(n / 2) - 1)  # !!tmp函数定义
#     tmp2 = iter_law_convolution(s_e1u, int(n / 2) - 1)
#     tmp = law_convolution(tmp1, tmp2)
#     # print(tmp)
#
#     sest_ave = [0] * n  # secret estimation using the original method of AGJ+19
#     sest_lar = [0] * n  # secret estimation using the largest value
#
#     tmp_e = law_convolution(tmp, s_e1u)
#     tmp_e = tail_probability(tmp_e)
#     # print("tmp_e", tmp_e)
#
#     for i in range(0, int(n / 2)):
#         fgivensci = []
#         for j in range(0, 2 * eta1 + 1):
#             pro_log = np.log(s_dict[j - eta1])
#             for k in range(0, m):
#                 p = tmp_e[0 - s_list[j] * a[k][i]]
#                 # print("p", p)
#                 # p = np.clip(p, 10e-15, None)
#                 pro_log += np.log(p)
#             fgivensci.append(pro_log)
#             # print("pro_log", pro_log)
#
#         # fgivensci = np.array(fgivensci) / sum(fgivensci)
#         # print("fgivensci", fgivensci)
#         max_log_prob = np.max(fgivensci)
#         stable_fgivensci = fgivensci - max_log_prob
#         # 将对数概率转换回普通概率
#         fgivensci = np.exp(stable_fgivensci)
#
#         # 归一化概率
#         fgivensci = fgivensci / np.sum(fgivensci)
#         # print(fgivensci)
#
#         for j in range(0, 2 * eta1 + 1):
#             # compute the secret key using the average value
#             sest_ave[i] += (j - eta1) * fgivensci[j]
#         # compute the secret key using the value with the largest prob
#         sest_lar[i] = np.argmax(fgivensci) - eta1
#
#     tmp_s = law_convolution(tmp, er)
#     tmp_s = tail_probability(tmp_s)
#     for i in range(int(n / 2), n):
#         fgivensci = []
#         for j in range(0, 2 * eta1 + 1):
#             pro_log = np.log(s_dict[j - eta1])
#             for k in range(0, m):
#                 p = tmp_s[0 - s_list[j] * a[k][i]]  # / tmp_s[qt + 1 - 50]
#                 # print("p", p)
#                 pro_log += np.log(p)
#             fgivensci.append(pro_log)
#             # fgivensci.append(np.exp(pro_log)),会导致数值下溢
#         # print("fgivensci", fgivensci)
#         max_log_prob = np.max(fgivensci)
#         stable_fgivensci = fgivensci - max_log_prob
#         # 将对数概率转换回普通概率
#         fgivensci = np.exp(stable_fgivensci)
#
#         # 归一化概率
#         fgivensci = fgivensci / np.sum(fgivensci)
#         # print(fgivensci)
#
#         for j in range(0, 2 * eta1 + 1):
#             # compute the secret key using the average value
#             sest_ave[i] += (j - eta1) * fgivensci[j]
#         # compute the secret key using the value with the largest prob
#         sest_lar[i] = np.argmax(fgivensci) - eta1
#
#     E_int_ave = [0] * n
#     for i in range(0, n):  # round numbers of secret
#         E_int_ave[i] = round(sest_ave[i])
#         if E_int_ave[i] > eta1:
#             E_int_ave[i] = eta1
#         elif E_int_ave[i] < -eta1:
#             E_int_ave[i] = -eta1
#     # print("sest_ave", E_int_ave)
#     # print("sest_lar", sest_lar)
#
#     matnum_ave = 0
#     matnum_lar = 0
#     for i in range(0, n):  # round numbers of secret
#         if E_int_ave[i] == solution[i]:
#             matnum_ave += 1
#         # if sest_lar[i] == solution[i]:
#         #     matnum_lar += 1
#
#     # print('DGJ+19, ciphertexts %d, experimental_matchnum using average value %d,'
#     #       ' experimental_matchnum using largest prob value %d' % (m, matnum_ave, matnum_lar))
#     print('DGJ+19, ciphertexts %d, experimental_matchnum using average value %d,' % (m, matnum_ave))
#
#     return E_int_ave, matnum_ave
#
#
# # 20241216
# # using the conditional probability method in DGJ19; using iter updating method, updating probability one by one
# def solve_inequalities_DGJ19_obo(eta, a, b, solution=None):
#     m, n = a.shape
#
#     s_list = []  # the list of secret s, e, r
#     for i in range(0, 2 * eta1 + 1):
#         s_list.append(i - eta1)
#     s_dict = {}
#     for i in range(0, 2 * eta1 + 1):
#         s_dict[i - eta1] = binom.pmf(i, 2 * eta1, 0.5)
#     # print("s_dict: ", s_dict)
#
#     e1_list = []  # the list of secret s, e, r
#     for i in range(0, 2 * eta2 + 1):
#         e1_list.append(i - eta2)
#     e1_dict = {}
#     for i in range(0, 2 * eta2 + 1):
#         e1_dict[i - eta2] = binom.pmf(i, 2 * eta2, 0.5)
#     # print("e1_dict: ", e1_dict)
#
#     # e1_u = law_convolution(e1_dict, e1_dict)
#     # print("e1_u: ", e1_u)
#
#     # make the Conditional probability distribution
#     er = law_product(s_dict, s_dict)
#     # print("er: ", er)
#
#     s_e1u = law_product(s_dict, e1_dict)
#     # print("s_e1u: ", s_e1u)
#
#     tmp1 = iter_law_convolution(er, int(n / 2) - 1)  # !!tmp函数定义
#     tmp2 = iter_law_convolution(s_e1u, int(n / 2) - 1)
#     tmp = law_convolution(tmp1, tmp2)
#     # print(tmp)
#
#     sest_ave = [0] * n  # secret estimation using the original method of AGJ+19
#     sest_lar = [0] * n  # secret estimation using the largest value
#
#     tmp_e = law_convolution(tmp, s_e1u)
#     tmp_e = tail_probability(tmp_e)
#     # print("tmp_e", tmp_e)
#
#     for i in range(0, int(n / 2)):
#         fgivensci = []
#         for j in range(0, 2 * eta1 + 1):
#             pro_log = np.log(s_dict[j - eta1])
#             for k in range(0, m):
#                 p = tmp_e[0 - s_list[j] * a[k][i]]
#                 # print("p", p)
#                 # p = np.clip(p, 10e-15, None)
#                 pro_log += np.log(p)
#             fgivensci.append(pro_log)
#             # print("pro_log", pro_log)
#
#         # fgivensci = np.array(fgivensci) / sum(fgivensci)
#         # print("fgivensci", fgivensci)
#         max_log_prob = np.max(fgivensci)
#         stable_fgivensci = fgivensci - max_log_prob
#         # 将对数概率转换回普通概率
#         fgivensci = np.exp(stable_fgivensci)
#
#         # 归一化概率
#         fgivensci = fgivensci / np.sum(fgivensci)
#         # print(fgivensci)
#
#         for j in range(0, 2 * eta1 + 1):
#             # compute the secret key using the average value
#             sest_ave[i] += (j - eta1) * fgivensci[j]
#         # compute the secret key using the value with the largest prob
#         sest_lar[i] = np.argmax(fgivensci) - eta1
#
#     tmp_s = law_convolution(tmp, er)
#     tmp_s = tail_probability(tmp_s)
#     for i in range(int(n / 2), n):
#         fgivensci = []
#         for j in range(0, 2 * eta1 + 1):
#             pro_log = np.log(s_dict[j - eta1])
#             for k in range(0, m):
#                 p = tmp_s[0 - s_list[j] * a[k][i]]  # / tmp_s[qt + 1 - 50]
#                 # print("p", p)
#                 pro_log += np.log(p)
#             fgivensci.append(pro_log)
#             # fgivensci.append(np.exp(pro_log)),会导致数值下溢
#         # print("fgivensci", fgivensci)
#         max_log_prob = np.max(fgivensci)
#         stable_fgivensci = fgivensci - max_log_prob
#         # 将对数概率转换回普通概率
#         fgivensci = np.exp(stable_fgivensci)
#
#         # 归一化概率
#         fgivensci = fgivensci / np.sum(fgivensci)
#         # print(fgivensci)
#
#         for j in range(0, 2 * eta1 + 1):
#             # compute the secret key using the average value
#             sest_ave[i] += (j - eta1) * fgivensci[j]
#         # compute the secret key using the value with the largest prob
#         sest_lar[i] = np.argmax(fgivensci) - eta1
#
#     E_int_ave = [0] * n
#     for i in range(0, n):  # round numbers of secret
#         E_int_ave[i] = round(sest_ave[i])
#         if E_int_ave[i] > eta1:
#             E_int_ave[i] = eta1
#         elif E_int_ave[i] < -eta1:
#             E_int_ave[i] = -eta1
#     # print("sest_ave", E_int_ave)
#     # print("sest_lar", sest_lar)
#
#     matnum_ave = 0
#     matnum_lar = 0
#     for i in range(0, n):  # round numbers of secret
#         if E_int_ave[i] == solution[i]:
#             matnum_ave += 1
#         # if sest_lar[i] == solution[i]:
#         #     matnum_lar += 1
#
#     # print('DGJ+19, ciphertexts %d, experimental_matchnum using average value %d,'
#     #       ' experimental_matchnum using largest prob value %d' % (m, matnum_ave, matnum_lar))
#     print('DGJ+19, ciphertexts %d, experimental_matchnum using average value %d,' % (m, matnum_ave))
#
#     return E_int_ave, matnum_ave
#
#
# # Secret Estimation in [DGJ+19]
# # estimate the secret using the conditional probability
# def solve_inequalities_Del22_2(eta1, eta2, a, b, is_geq_zero, solution=None):
#     eta1 = eta1
#     eta2 = eta2
#     m, n = a.shape
#
#     s_list = []  # the list of secret s, e, r
#     for i in range(0, 2 * eta1 + 1):
#         s_list.append(i - eta1)
#     s_dict = {}
#     for i in range(0, 2 * eta1 + 1):
#         s_dict[i - eta1] = 1 / (2 * eta1 + 1)
#     # print("s_dict: ", s_dict)
#
#     e1_list = []  # the list of secret s, e, r
#     for i in range(0, 2 * eta2 + 1):
#         e1_list.append(i - eta2)
#     e1_dict = {}
#     for i in range(0, 2 * eta2 + 1):
#         e1_dict[i - eta2] = 1 / (2 * eta2 + 1)
#     # print("e1_dict: ", e1_dict)
#
#     e1_u = law_convolution(e1_dict, e1_dict)
#     # print("e1_u: ", e1_u)
#
#     # make the Conditional probability distribution
#     er = law_product(s_dict, s_dict)
#     # print("er: ", er)
#
#     s_e1u = law_product(s_dict, e1_u)
#     # print("s_e1u: ", s_e1u)
#
#     tmp1 = iter_law_convolution(er, int(n / 2) - 1)  # !!tmp函数定义
#     tmp2 = iter_law_convolution(s_e1u, int(n / 2) - 1)
#     tmp = law_convolution(tmp1, tmp2)
#     # tmp_full = law_convolution(iter_law_convolution(er, int(n/2)), iter_law_convolution(s_e1u, int(n/2)))
#     # print(tmp)
#
#     # generate random failures and calculate their angle with the secret
#
#     failure = []
#     b = b
#     # as > b
#     for i in range(0, m):
#         if is_geq_zero[i]:
#             df = np.array(a[i])
#             b[i] = b[i]
#         else:
#             df = np.array(-a[i])
#             b[i] = -b[i]
#         failure.append(df)
#     # print("b: ", b)
#
#     sest_ave = [0] * n  # secret estimation using the original method of AGJ+19
#     sest_lar = [0] * n  # secret estimation using the largest value
#
#     tmp_e = law_convolution(tmp, s_e1u)
#     tmp_e = tail_probability(tmp_e)
#     # print("tmp_e", tmp_e)
#
#     for i in range(0, int(n / 2)):
#         fgivensci = []
#         for j in range(0, 2 * eta1 + 1):
#             pro_log = 0
#             for k in range(0, m):
#                 p = 1 - tmp_e[b[k] + s_list[j] * failure[k][i]]
#                 # print("p", p)
#                 pro_log += np.log(p)
#             fgivensci.append(pro_log)
#             # print("pro_log", pro_log)
#
#         # fgivensci = np.array(fgivensci) / sum(fgivensci)
#         # print("fgivensci", fgivensci)
#         max_log_prob = np.max(fgivensci)
#         stable_fgivensci = fgivensci - max_log_prob
#         # 将对数概率转换回普通概率
#         fgivensci = np.exp(stable_fgivensci)
#
#         # 归一化概率
#         fgivensci = fgivensci / np.sum(fgivensci)
#         # print(fgivensci)
#
#         for j in range(0, 2 * eta1 + 1):
#             # compute the secret key using the average value
#             sest_ave[i] += (j - eta1) * fgivensci[j]
#         # compute the secret key using the value with the largest prob
#         sest_lar[i] = np.argmax(fgivensci) - eta1
#
#     tmp_s = law_convolution(tmp, er)
#     tmp_s = tail_probability(tmp_s)
#     for i in range(int(n / 2), n):
#         fgivensci = []
#         for j in range(0, 2 * eta1 + 1):
#             pro_log = 0
#             for k in range(0, m):
#                 p = 1 - tmp_s[b[k] + s_list[j] * failure[k][i]]
#                 # print("p", p)
#                 pro_log += np.log(p)
#             fgivensci.append(pro_log)
#             # fgivensci.append(np.exp(pro_log)),会导致数值下溢
#         # print("fgivensci", fgivensci)
#         max_log_prob = np.max(fgivensci)
#         stable_fgivensci = fgivensci - max_log_prob
#         # 将对数概率转换回普通概率
#         fgivensci = np.exp(stable_fgivensci)
#
#         # 归一化概率
#         fgivensci = fgivensci / np.sum(fgivensci)
#         # print(fgivensci)
#
#         for j in range(0, 2 * eta1 + 1):
#             # compute the secret key using the average value
#             sest_ave[i] += (j - eta1) * fgivensci[j]
#         # compute the secret key using the value with the largest prob
#         sest_lar[i] = np.argmax(fgivensci) - eta1
#
#     E_int_ave = [0] * n
#     for i in range(0, n):  # round numbers of secret
#         E_int_ave[i] = round(sest_ave[i])
#         if E_int_ave[i] > eta1:
#             E_int_ave[i] = eta1
#         elif E_int_ave[i] < -eta1:
#             E_int_ave[i] = -eta1
#     # print("sest_ave", E_int_ave)
#     print("sest_lar", sest_lar)
#
#     matnum_ave = 0
#     matnum_lar = 0
#     for i in range(0, n):  # round numbers of secret
#         if E_int_ave[i] == solution[i]:
#             matnum_ave += 1
#         if sest_lar[i] == solution[i]:
#             matnum_lar += 1
#
#     print('DGJ+19_2, ciphertexts %d, experimental_matchnum using average value %d,'
#           ' experimental_matchnum using largest prob value %d' % (m, matnum_ave, matnum_lar))
#
#
# # solving the secret only perfect hints of Kyber using the method of DGJ+19
# def solve_perfect_hints_DGJ19(eta, a, b, solution=None, so_flag=None):
#     if so_flag:
#         print("Solving secret only perfect hints...")
#     else:
#         print("Solving secret error perfect hints...")
#     eta = eta
#     [nb_of_hints, nb_of_unknowns] = a.shape
#     guess = np.zeros((nb_of_unknowns), dtype=int)  # creat an initial guess of the solution with all values set to zero
#     if nb_of_hints == 0:
#         return guess
#
#     v_01_dict = {0: 0.5, 1: 0.5}
#
#     q = 3329
#     v_q_list = []
#     for i in range(q):
#         v_q_list.append(i - int((q - 1) / 2))
#     v_q_dict = {}
#     for i in range(q):
#         v_q_dict[i - int((q - 1) / 2)] = 1 / q
#
#     v_cbd_dict = {-3: 1 / 64, -2: 6 / 64, -1: 15 / 64, 0: 20 / 64, 1: 15 / 64, 2: 6 / 64, 3: 1 / 64}
#
#     # s_dict = {-3:1/64, -2:6/64, -1:15/64, 0:20/64, 1:15/64, 2:6/64, 3:1/64}
#     # s_dict = {-3:0.015625, -2:0.09375, -1:0.234375, 0:0.3125, 1:0.234375, 2:0.09375, 3:0.015625}
#
#     s_list = []
#     for i in range(0, 2 * eta + 1):
#         s_list.append(i - eta)
#     s_dict = {}
#     for i in range(0, 2 * eta + 1):
#         s_dict[i - eta] = binom.pmf(i, 2 * eta, 0.5)
#     print("s_dict: ", s_dict)
#
#     vs = law_product(v_cbd_dict, s_dict)
#     # print("vs_dict: ", vs)
#
#     tmp = iter_law_convolution(vs, nb_of_unknowns - 1)  # !!tmp函数定义
#     print("tmp", tmp)
#
#     sest_ave = [0] * nb_of_unknowns  # secret estimation using the original method of AGJ+19
#     sest_lar = [0] * nb_of_unknowns  # secret estimation using the largest value
#
#     for i in range(nb_of_unknowns):
#         fgivensci = []
#         for j in range(0, 2 * eta + 1):
#             pro_log = np.log(s_dict[j - eta])  # 初始概率分布
#             for k in range(0, nb_of_hints):
#                 p = tmp[b[k] - s_list[j] * a[k][i]]
#                 # p = np.clip(p, 10e-15, None)
#                 pro_log += np.log(p)
#             fgivensci.append(pro_log)
#             # print("pro_log", pro_log)
#
#         max_log_prob = np.max(fgivensci)
#         stable_fgivensci = fgivensci - max_log_prob
#         # 将对数概率转换回普通概率
#         fgivensci = np.exp(stable_fgivensci)
#         # 归一化概率
#         fgivensci = fgivensci / np.sum(fgivensci)
#         if i < 5:
#             print("fgivensci", fgivensci)
#
#         for j in range(0, 2 * eta + 1):
#             # compute the secret key using the average value
#             sest_ave[i] += (j - eta) * fgivensci[j]
#         # compute the secret key using the value with the largest prob
#         sest_lar[i] = np.argmax(fgivensci) - eta
#
#     E_int_ave = [0] * nb_of_unknowns
#     for i in range(0, nb_of_unknowns):  # round numbers of secret
#         E_int_ave[i] = round(sest_ave[i])
#         if E_int_ave[i] > eta:
#             E_int_ave[i] = eta
#         elif E_int_ave[i] < -eta:
#             E_int_ave[i] = -eta
#     print("sest_ave", E_int_ave)
#     print("sest_lar", sest_lar)
#
#     matnum_ave = 0
#     matnum_lar = 0
#     for i in range(0, nb_of_unknowns):  # round numbers of secret
#         if E_int_ave[i] == solution[i]:
#             matnum_ave += 1
#         if sest_lar[i] == solution[i]:
#             matnum_lar += 1
#
#     print('DGJ+19, number of perfect hints %d, experimental_matchnum using average value %d,'
#           ' experimental_matchnum using largest prob value %d' % (nb_of_hints, matnum_ave, matnum_lar))
#     short_vector = np.concatenate((np.array(E_int_ave - solution), np.array([1])))
#     distance = np.linalg.norm(short_vector)
#     distance = np.round(distance, 2)
#     print("distance", distance)
#     return E_int_ave, matnum_ave, distance
#
#
# # solving the perfect hints of Kyber using the method of DGJ+19，simulate the distribution using CLT
# def solve_perfect_hints_DGJ19_CLT(eta, a, b, solution=None, so_flag=None):
#     if so_flag:
#         print("Solving secret only perfect hints...")
#     else:
#         print("Solving secret error perfect hints...")
#     [nb_of_hints, nb_of_unknowns] = a.shape
#     print("the number of hints", nb_of_hints)
#     guess = np.zeros(nb_of_unknowns, dtype=int)  # creat an initial guess of the solution with all values set to zero
#     if nb_of_hints == 0:
#         print("the number of hints is 0 !")
#         return guess
#
#     eta = eta
#     nb_of_values = 2 * eta + 1
#     x = np.arange(-eta, eta + 1, dtype=np.int8)
#     x_pmf = binom.pmf(x + eta, 2 * eta, 0.5)
#     # print("x_pmf", x_pmf)
#     x_pmf = np.repeat(x_pmf.reshape(1, -1), nb_of_unknowns,
#                       axis=0)  # this line repeats the x_pmf array multiple times to creat a 2D array
#     a = a.astype(np.int16)  # this change the datatype of the matrix a to int16
#     a_squared = np.square(a)  # this squares each element of a
#
#     mean = np.matmul(x_pmf, x)  # 计算当前分布下，所有未知数的期望值
#     variance = np.matmul(x_pmf, np.square(x)) - np.square(mean)  # 方差计算公式
#     mean = np.multiply(a, np.repeat(mean[np.newaxis, :], nb_of_hints, axis=0))
#     variance = np.multiply(a_squared, np.repeat(variance[np.newaxis, :], nb_of_hints, axis=0))
#     mean = mean.sum(axis=1).reshape(-1, 1).repeat(nb_of_unknowns, axis=1) - mean
#     # print("mean", mean)
#     mean = -b[:, np.newaxis]
#     # print("mean", mean)
#     # mean += qt # 实验表明，750， 800效果最佳
#     variance = variance.sum(axis=1).reshape(-1, 1).repeat(nb_of_unknowns, axis=1) - variance
#     # print("variance", variance)
#     variance = np.clip(variance, 1, None)
#     psuccess = np.zeros((nb_of_values, nb_of_hints, nb_of_unknowns), dtype=float)
#     for j in range(nb_of_values):
#         zscore = np.divide(a * x[j] + mean + 0.5, np.sqrt(variance))
#         # psuccess[j, :, :] = norm.cdf(zscore) / norm.cdf(np.divide(mean + 0.5, np.sqrt(variance))) # central limit theorem
#         psuccess[j, :, :] = norm.cdf(zscore + 0.5) - norm.cdf(zscore - 0.5)
#         print("psuccess[j, 0, 0]", psuccess[j, 0, 0])
#     psuccess = np.transpose(psuccess, axes=[2, 0, 1])
#     psuccess = np.sum(np.log(psuccess), axis=2)
#     # print("psuccess", psuccess)
#     psuccess = np.exp(psuccess)
#     # psuccess = np.sum(psuccess)
#
#     x_pmf = np.multiply(psuccess, x_pmf)
#     # print("x_pmf", x_pmf)
#     row_sums = x_pmf.sum(axis=1)
#     x_pmf /= row_sums[:, np.newaxis]
#
#     # for i in range(3):
#     #     print("x_pmf%d" %i)
#     #     for element in x_pmf[i]:
#     #         print(f"{element:.16f}")
#     # guess_ave = np.matmul(x_pmf, x)
#     # print("guess_ave", guess_ave)
#
#     guess = x[np.argmax(x_pmf, axis=1)]
#     print("guess ", guess)
#
#     nb_correct = 0
#     if solution is not None:
#         nb_correct = np.count_nonzero(solution == guess)
#         print("Number of correctly guessed unknowns: {:d}/{:d}".format(nb_correct, len(solution)))
#
#     short_vector = np.concatenate((np.array(guess - solution), np.array([1])))
#     distance = np.linalg.norm(short_vector)
#     distance = np.round(distance, 2)
#     print("distance", distance)
#     return guess, nb_correct, distance
#
#
# # solving the secret only perfect hints of Kyber using the method of del22
# # using the CLT to simulate the distribution of \Sigma v_i*s_i
# # Eliminate the failure rate
# def solve_perfect_hints_Del22(eta, a, b, max_nb_of_iterations=10, solution=None, so_flag=None):
#     if so_flag:
#         print("Solving secret only perfect hints...")
#     else:
#         print("Solving secret error perfect hints...")
#
#     eta = eta
#     [nb_of_hints, nb_of_unknowns] = a.shape
#     guess = np.zeros((nb_of_unknowns), dtype=int)  # creat an initial guess of the solution with all values set to zero
#     if nb_of_hints == 0:
#         return guess
#     nb_of_values = 2 * eta + 1
#     x = np.arange(-eta, eta + 1, dtype=np.int8)
#     x_pmf = binom.pmf(x + eta, 2 * eta, 0.5)
#     print("x_pmf: ", x_pmf)
#     x_pmf = np.repeat(x_pmf.reshape(1, -1), nb_of_unknowns,
#                       axis=0)  # this line repeats the x_pmf array multiple times to creat a 2D array
#
#     a = a.astype(np.int16)  # this change the datatype of the matrix a to int16
#     a_squared = np.square(a)  # this squares each element of a
#
#     count = [0] * max_nb_of_iterations
#     for z in range(max_nb_of_iterations):
#         print("Iteration " + str(z))
#         time_start = time.time()
#         mean = np.matmul(x_pmf, x)  # 计算当前分布下，所有未知数的期望值
#         variance = np.matmul(x_pmf, np.square(x)) - np.square(mean)
#         mean = np.multiply(a, np.repeat(mean[np.newaxis, :], nb_of_hints, axis=0))
#         variance = np.multiply(a_squared, np.repeat(variance[np.newaxis, :], nb_of_hints, axis=0))
#         mean = mean.sum(axis=1).reshape(-1, 1).repeat(nb_of_unknowns, axis=1) - mean  # 减去自身
#         # print("mean",mean[:3])
#         mean -= b[:, np.newaxis]
#         # print("mean", mean[:3])
#         variance = variance.sum(axis=1).reshape(-1, 1).repeat(nb_of_unknowns, axis=1) - variance
#         variance = np.clip(variance, 1, None)
#         psuccess = np.zeros((nb_of_values, nb_of_hints, nb_of_unknowns), dtype=float)
#         for j in range(nb_of_values):
#             # 求解连续高斯分布中某个点值的概率
#             # zscore_pos = np.divide(a * x[j] + mean + 0.5 + 100, np.sqrt(variance))
#             # zscore_neg = np.divide(a * x[j] + mean + 0.5 - 100, np.sqrt(variance))
#             # psuccess[j, :, :] = norm.cdf(zscore_pos) - norm.cdf(zscore_neg)  # central limit theorem
#             # SMY：连续分布无法计算d=zscore的概率，使用cdf(zscore+1)-cdf(zscore-1)
#             zscore = np.divide(a * x[j] + mean + 0.5, np.sqrt(variance))
#             # print("zscore", zscore)
#             psuccess[j, :, :] = norm.cdf(zscore + 0.5) - norm.cdf(zscore - 0.5)  # Kyber128:0.5; Kyber256:1; Kyber512。
#         # print("psuccess[, 0, 0]",psuccess[:, 0, 0])
#
#         psuccess = np.transpose(psuccess, axes=[2, 0, 1])
#         # print("psuccess", psuccess[0])
#         psuccess = np.clip(psuccess, 10e-20, None)
#         psuccess = np.sum(np.log(psuccess), axis=2)
#         # print("psuccess", psuccess)
#         row_means = psuccess.max(axis=1)
#         psuccess -= row_means[:, np.newaxis]
#         # print("row_means",row_means)
#         psuccess = np.exp(psuccess)
#
#         x_pmf = np.multiply(psuccess, x_pmf)
#         row_sums = x_pmf.sum(axis=1)
#         x_pmf /= row_sums[:, np.newaxis]
#         guess = x[np.argmax(x_pmf, axis=1)]
#
#         if z == max_nb_of_iterations - 1:
#             print(np.array(guess))
#
#         time_end = time.time()
#         print("Elapsed time: {:.1f} seconds".format(time_end - time_start))
#         if solution is not None:
#             nb_correct = np.count_nonzero(solution == guess)
#             count[z] = nb_correct
#             print("Number of correctly guessed unknowns: {:d}/{:d}"
#                   .format(nb_correct, len(solution)))
#         # if (z > 1) and count[z-1] >= count[z] + 1:
#         #     # print(np.array(guess))
#         #     count[z] = count[z - 1]
#         #
#         #     break
#     print("count", count)
#     print("guess", np.array(guess))
#     short_vector = np.concatenate((np.array(guess - solution), np.array([1])))
#     distance = np.linalg.norm(short_vector)
#     distance = np.round(distance, 2)
#     print("distance", distance)
#     return guess, nb_correct, distance
#
#
# # solving the translated secret only perfect hints of Kyber using the method of del22
# # using the CLT to simulate the distribution of \Sigma v_i*s_i
# # Eliminate the failure rate
# def solve_ineq_perfect_hints_del22(eta, a, b, is_geq_zero, max_nb_of_iterations=20, solution=None):
#     print("Solving translated secret only perfect hints...")
#
#     [nb_of_hints, nb_of_unknowns] = a.shape
#     guess = np.zeros(nb_of_unknowns, dtype=int)  # creat an initial guess of the solution with all values set to zero
#
#     if nb_of_hints == 0:
#         print("the number of hints is 0 !")
#         return guess
#     nb_of_values = 2 * eta + 1
#     x = np.arange(-eta, eta + 1, dtype=np.int8)
#     x_pmf = binom.pmf(x + eta, 2 * eta, 0.5)
#     print("x_pmf", x_pmf)
#     x_pmf = np.repeat(x_pmf.reshape(1, -1), nb_of_unknowns, axis=0)
#     # x_pmf_static = x_pmf.copy()
#     a = a.astype(np.int16)
#     a_squared = np.square(a)
#
#     count = [0] * max_nb_of_iterations
#     for z in range(max_nb_of_iterations):
#         print("Iteration " + str(z))
#         print("**x_pmf", x_pmf[0, :])
#         time_start = time.time()
#         mean = np.matmul(x_pmf, x)  # 计算当前分布下，所有未知数的期望值
#         variance = np.matmul(x_pmf, np.square(x)) - np.square(mean)
#         mean = np.multiply(a, np.repeat(mean[np.newaxis, :], nb_of_hints, axis=0))
#         variance = np.multiply(a_squared, np.repeat(variance[np.newaxis, :], nb_of_hints, axis=0))
#         mean = mean.sum(axis=1).reshape(-1, 1).repeat(nb_of_unknowns, axis=1) - mean  # 减去自身
#         print("mean", mean[:3, :3])
#         mean -= b[:, np.newaxis]
#         print("mean", mean[:3, :3])
#         variance = variance.sum(axis=1).reshape(-1, 1).repeat(nb_of_unknowns, axis=1) - variance
#         variance = np.clip(variance, 1, None)
#         psuccess = np.zeros((nb_of_values, nb_of_hints, nb_of_unknowns), dtype=float)
#         for j in range(nb_of_values):
#             zscore = np.divide(a * x[j] + mean + 0.5, np.sqrt(variance))
#             psuccess[j, :, :] = norm.cdf(zscore)  # central limit theorem
#         print("psuccess[, 0, :5]", psuccess[:, 0, :5])
#
#         psuccess = np.transpose(psuccess, axes=[2, 0, 1])
#         psuccess = np.multiply(psuccess, is_geq_zero) + np.multiply(1 - psuccess, 1 - is_geq_zero)
#         psuccess = np.clip(psuccess, 10e-5, None)
#         psuccess = np.sum(np.log(psuccess), axis=2)
#         row_means = psuccess.max(axis=1)
#         psuccess -= row_means[:, np.newaxis]
#         psuccess = np.exp(psuccess)
#
#         # x_pmf = np.multiply(psuccess, x_pmf_static) # SMY:20241009
#         x_pmf = np.multiply(psuccess, x_pmf)  # SMY:20241009
#         # print("x_pmf_before_nor", x_pmf[0, :])
#         row_sums = x_pmf.sum(axis=1)
#         x_pmf /= row_sums[:, np.newaxis]
#         print("x_pmf", x_pmf[:5, :])
#         guess = x[np.argmax(x_pmf, axis=1)]
#         print("guess", np.array(guess))
#
#         if z == max_nb_of_iterations - 1:
#             print(np.array(guess))
#
#         time_end = time.time()
#         print("Elapsed time: {:.1f} seconds".format(time_end - time_start))
#
#         if solution is not None:
#             nb_correct = np.count_nonzero(solution == guess)
#             count[z] = nb_correct
#             print("Number of correctly guessed unknowns: {:d}/{:d}".format(nb_correct, len(solution)))
#
#         if (z > 1) and count[z - 1] >= count[z] + 2:
#             # print(np.array(guess))
#             count[z] = count[z - 1]
#             break
#     print("count", count)
#     print("guess", np.array(guess))
#     short_vector = np.concatenate((np.array(guess - solution), np.array([1])))
#     distance = np.linalg.norm(short_vector)
#     distance = np.round(distance, 2)
#     print("distance", distance)
#     return guess, nb_correct, distance
#
#
# # 中心极限定理偏差太大，迭代后均值太大
#
#
# def test_inequalities():
#     rng = RNG_OS()
#     nb_of_inequalities = 1000
#     # for K in [2, 3, 4]:
#     for K in [2]:
#         kem = KyberKEM(K, rng)
#         [public_key, private_key, s, e] = kem.keygen(return_internals=True)
#         solution = np.concatenate((s.ravel(), e.ravel()))
#         [a, b, manipulated_indices, _, manipulated_ciphertexts, _,
#          manipulated_shared_secrets] = generate_inequalities(kem,
#                                                              public_key, nb_of_inequalities, return_manipulation=True)
#         is_geq_zero1 = evaluate_inequalities_slow(kem, private_key,
#                                                   manipulated_indices, manipulated_ciphertexts,
#                                                   manipulated_shared_secrets)
#         is_geq_zero2 = evaluate_inequalities_fast(a, b, solution)
#         if np.any(is_geq_zero1 != is_geq_zero2):
#             raise ValueError("Inequalities test failed")
#         print("Inequalities test passed")
#
#
# def test_equalities():
#     rng = RNG_OS()
#     # for K in [2, 3, 4]:
#     for K in [2]:
#         kem = KyberKEM(K, rng)
#         [public_key, private_key, s, e] = kem.keygen(return_internals=True)
#         solution = np.concatenate((s.ravel(), e.ravel()))
#         [a, b] = generate_equalities(kem, public_key)
#         r = (np.matmul(a, solution) + b) % KyberPKE.Q
#         if r.any():
#             raise ValueError("Equalities test failed")
#         print("Equalities test passed")
#
#
# def test():
#     print("Testing solver...")
#     test_inequalities()
#     test_equalities()
#
#
# def normalize(vector):
#     if np.linalg.norm(vector) == 0:
#         return vector
#     else:
#         return vector / np.linalg.norm(vector)
#
#
# # define the uniform distribution around one target value with [-3,3]
# def tar_uniform(eta, k):
#     x = np.arange(-eta, eta + 1, dtype=np.int8)
#     x_pmf = [0] * (2 * eta + 1)
#     if k == -eta:
#         x_pmf[k + eta] = x_pmf[k + eta + 1] = 1 / 2
#     elif k == eta:
#         x_pmf[k + eta] = x_pmf[k + eta - 1] = 1 / 2
#     else:
#         x_pmf[k + eta - 1] = x_pmf[k + eta] = x_pmf[k + eta + 1] = 1 / 3
#     return x_pmf