#!/usr/bin/python3
import random
import numpy as np
from scipy import stats
from tqdm import tqdm
# from major_vote import major_vote_any, major_vote_any_512


def normalize(vector):
    if np.linalg.norm(vector) == 0:
        return vector
    else:
        return vector / np.linalg.norm(vector)


def major_vote_any(a, solution):
    a = np.array(a)
    # 初始化众数数组和众数出现次数
    guess_mode = np.zeros(len(a[0]), dtype=int)
    counts = np.zeros(len(a[0]), dtype=int)

    # 计算众数和众数出现次数
    for i in range(len(a[0])):
        mode_info = stats.mode(a[:,i])
        guess_mode[i] = mode_info.mode

    anglelist_est = np.dot(solution, guess_mode) / (np.linalg.norm(solution)*np.linalg.norm(guess_mode))
    nb_correct = np.count_nonzero(solution == guess_mode)
    # print("Number of correctly guessed unknowns: {:d}/{:d}".format(nb_correct, 1024))

    return nb_correct, anglelist_est


def major_vote_any_512(a, solution):
    a = np.array(a)
    [nb_of_inequalities, nb_of_unknowns] = a.shape
    # 初始化众数数组和众数出现次数
    guess_mode = np.zeros(nb_of_unknowns, dtype=int)
    counts = np.zeros(nb_of_unknowns, dtype=int)

    # 计算众数和众数出现次数
    for i in range(nb_of_unknowns):
        mode_info = stats.mode(a[:,i])
        guess_mode[i] = mode_info.mode
        counts[i] = mode_info.count
    # print("众数数组：", guess_mode)
    # print("众数出现次数：", counts)

    nb_correct = np.count_nonzero(solution == guess_mode)
    # print("Number of correctly guessed unknowns: {:d}/{:d}".format(nb_correct, nb_of_unknowns))
    # print("guess:", np.array(guess_mode))


    # 对数组进行降序排序
    sorted_counts = np.argsort(counts)[::-1]
    # 选择前512个最大元素的索引
    half = int(nb_of_unknowns/2)
    index_pro = sorted_counts[:half]
    # 使用index_pro
    selected_guess = guess_mode[index_pro]
    selected_solution = solution[index_pro]
    matches = selected_guess == selected_solution
    num_of_matches = np.count_nonzero(matches)
    # print("Number of selected coeffs matches:{:d}/{:d}".format(num_of_matches, half))

    return num_of_matches


# estimate the secret key using these dfs on average
def solve_inequalities_average(eta, a, solution=None): # analyze convergence rate with a known solution
    # print("estimate the secret key using these dfs on average in DRV20")
    qt = 3329 / 4
    [nb_of_inequalities, nb_of_unknowns] = a.shape
    # print("nb_of_inequalities",nb_of_inequalities, "nb_of_unknowns", nb_of_unknowns)
    guess = np.zeros(nb_of_unknowns, dtype=int)
    failure_sum = [0] * nb_of_unknowns
    failure = []
    for i in range(0, nb_of_inequalities):
        df = np.array(a[i])
        failure.append(df)
        failure_sum += np.array(df)
    E = [0] * nb_of_unknowns
    for i in range(0, nb_of_inequalities):
        E += np.array(failure[i] / np.linalg.norm(failure[i]))

    sest_norm = nb_of_inequalities * qt / np.linalg.norm(failure_sum)
    E = np.array(normalize(E)) * sest_norm
    E_int = [0] * nb_of_unknowns
    for i in range(0, nb_of_unknowns):  # round numbers of secret
        E_int[i] = round(E[i])

        if E_int[i] >= eta:
            E_int[i] = eta
        elif E_int[i] <= -eta:
            E_int[i] = -eta

    E_int = np.array(E_int)
    # print("E_int", E_int)
    # print("E_int", solution)
    if solution is not None:
        nb_correct = np.count_nonzero(solution == E_int)
        # print("Number of correctly guessed unknowns by ARV20: {:d}/{:d}".format(nb_correct, len(solution)))
    return E_int, nb_correct


# decryption failure attack on dfs in DRV20
def df_DRV20(m, k, solution):
    """Solves one system of inequalities and tracks the convergence rate"""
    # q = 8192
    q = 3329
    eta = 2
    n_secret = int(len(solution)/2)
    nb_of_inequalities = 5000

    with open("DATA/DFA/Kyber1024/V.txt", 'r') as f:
        lines_a = [next(f) for _ in range(nb_of_inequalities)]
    a = np.loadtxt(lines_a)

    nb_of_unknowns = len(solution)
    print("nb_of_unknowns", nb_of_unknowns)

    if m == 0:
        E_int = [0] * nb_of_unknowns
        nb_correct = np.count_nonzero(solution == E_int)
        norm_value = np.linalg.norm(solution - E_int)
        print("Norm of (a - b):", norm_value)
        print("the average recovered coefficients with %d ineqs is %d" % (m, nb_correct))
        return nb_correct

    rec_num = []
    rec_num_major = []
    rec_num_major_half = []

    # calculate theta_SE approximately
    distance = []
    anglelist_est = []
    anglelist_est_major = []

    num_correct = 0
    num_correct_major = 0
    num_correct_major_half = 0

    for i in range(k):
        # 选择m个索引
        indices = random.sample(range(nb_of_inequalities), m)
        a_selected = np.array([a[j] for j in indices])

        # 恢复全部私钥
        s, n = solve_inequalities_average(eta, a_selected, solution=solution)
        dis = np.linalg.norm(solution - s)
        distance.append(dis)
        cos = np.dot(solution, s) / (np.linalg.norm(solution)*np.linalg.norm(s))
        # print("cos = ", cos)
        anglelist_est.append(cos)

        rec_num.append(n)
        if n == nb_of_unknowns:
            num_correct += 1
        # print("Number of selected coeffs matches:{:d}/{:d}".format(n, nb_of_unknowns))

        # 使用major vote 恢复私钥
        guess_block = []
        for i in range(23):
            # 选择m/2个索引
            indices_major = random.sample(range(m), int(1*m / 2))
            a_selected_major = np.array([a_selected[j] for j in indices_major])
            s_major, n_major = solve_inequalities_average(eta, a_selected_major, solution=solution)
            guess_block.append(s_major)

        # 使用major vote 恢复私钥
        n_major_mode, anglelist_major = major_vote_any(guess_block, solution)
        rec_num_major.append(n_major_mode)
        anglelist_est_major.append(anglelist_major)
        if n_major_mode == len(solution):
            num_correct_major += 1

        # 使用major vote 恢复一半私钥
        n_major_mode_half = major_vote_any_512(guess_block, solution)
        rec_num_major_half.append(n_major_mode_half)
        if n_major_mode_half == n_secret:
            num_correct_major_half += 1
        # print("Number of selected coeffs matches:{:d}/{:d}".format(n_major_mode, int(nb_of_unknowns/2)))

    ave_rec_num = np.mean(rec_num)
    ave_rec_dis = np.mean(distance)
    ave_angle = np.mean(anglelist_est)
    ave_rec_num_major_half = np.mean(rec_num_major_half)

    ave_rec_num_major = np.mean(rec_num_major)
    ave_angle_major = np.mean(anglelist_est_major)

    success_rate = num_correct / k
    success_rate_major = num_correct_major / k
    success_rate_major_half = num_correct_major_half / k

    print("the average recovered coefficients with %d ineqs is %f/%d" % (m, ave_rec_num, 2*n_secret))
    print("the average recovered distance with %d ineqs is %f" % (m, ave_rec_dis))
    print("the average cos(angle) of recovered coefficients is %f" % ave_angle)
    print("the success prob of recovering full ineqs with %d ineqs is %f" % (m, success_rate))

    print("major, the average recovered coefficients with %d ineqs is %f/%d" % (m, ave_rec_num_major, 2*n_secret))
    print("major, the average cos(angle) of recovered coefficients is %f" % ave_angle_major)
    print("major, the success prob of recovering full ineqs with %d ineqs is %f" % (m, success_rate_major))

    print("major_half, the average recovered coefficients with %d ineqs is %f/%d" % (m, ave_rec_num_major_half, n_secret))
    print("major_half, the success prob of recovering full ineqs with %d ineqs is %f" % (m, success_rate_major_half))

    return ave_rec_num, ave_rec_dis, ave_angle, success_rate, ave_rec_num_major, ave_angle_major, success_rate_major, ave_rec_num_major_half, success_rate_major_half


if __name__ == "__main__":

    solution = []
    with open("DATA/DFA/Kyber1024/es.txt", 'r') as g:
        for line in g:
            solution.append(int(line.strip()))
    solution = np.array(solution)

    num_ine = []

    num_rec = []
    dis_rec = []
    suc_rat = []
    cos_angle = []

    num_rec_major = []
    suc_rat_major = []
    cos_angle_major = []

    num_rec_major_half = []
    suc_rat_major_half = []

    for m in tqdm(range(20, 311, 10)):
        num_ine.append(m)
        print("The number of equalities is", m)
        rec, dis, angle, ratio, rec_major, angle_major, ratio_major, rec_major_half, ratio_major_half = df_DRV20(m, 10, solution)
        num_rec.append(rec)
        dis_rec.append(dis)
        cos_angle.append(angle)
        suc_rat.append(ratio)

        num_rec_major.append(rec_major)
        cos_angle_major.append(angle_major)
        suc_rat_major.append(ratio_major)

        num_rec_major_half.append(rec_major_half)
        suc_rat_major_half.append(ratio_major_half)

    print("num_ine: ", num_ine)

    print("num_rec: ", num_rec)
    print("dis_rec: ", dis_rec)
    print("cos_angle: ", cos_angle)
    print("suc_rat: ", suc_rat)

    print("num_rec_major: ", num_rec_major)
    print("cos_angle_major: ", cos_angle_major)
    print("suc_rat_major: ", suc_rat_major)

    print("num_rec_major_half: ", num_rec_major_half)
    print("suc_rat_major_half: ", suc_rat_major_half)

