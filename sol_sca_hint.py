import random
import solver
import numpy as np
from tqdm import tqdm


def sol_approx_hints(m, k, solution):
    ETA = 2
    q = 3329
    bias = int(q/64)  # q/32 for Kyber512 and 768； q/64 for Kyber1024

    nb_of_hints = 10000
    nb_of_unknowns = len(solution)
    # print("The number of unknowns", nb_of_unknowns)

    with open("Data/SCA/Kyber768/v.txt", 'r') as f:
        lines_v = [next(f) for _ in range(nb_of_hints)]
    V = np.loadtxt(lines_v)

    with open("Data/SCA/Kyber768/l.txt", 'r') as g:
        lines_l = [next(g) for _ in range(nb_of_hints)]
    L = np.loadtxt(lines_l)
    # print(b)

    if m == 0:
        E_int = [0] * nb_of_unknowns
        nb_correct = np.count_nonzero(solution == E_int)
        print("The average recovered coefficients with %d approximate hints is %d" % (m, nb_correct))
        # short_vector = np.concatenate((np.array(E_int - solution), np.array([1])))
        short_vector = np.array(E_int - solution)
        distance = np.linalg.norm(short_vector)
        distance = np.round(distance, 2)
        print("The average distance with %d approximate hints is %f" % (m, distance))
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

        # 恢复私钥
        s, n, d = solver.solve_sca_approx_hints_Del22(ETA, bias, V_selected, L_selected, solution=solution, so_flag=None)
        rec_num.append(n)
        rec_dis.append(d)
        if n == nb_of_unknowns:
            num_correct += 1
        # print("Number of selected coeffs matches:{:d}/{:d}".format(n, nb_of_unknowns))

    ave_rec_num = np.mean(rec_num)
    ave_rec_dis = np.round(np.mean(rec_dis), 2)
    success_rate = num_correct / k

    print("The average recovered coefficients with %d sca approximate hints is %f/%d" % (m, ave_rec_num, nb_of_unknowns))
    print("The average recovered distances with %d sca approximate hints is %f/%d" % (m, ave_rec_dis, nb_of_unknowns))
    print("The success prob of recovering full coes with %d sca approximate hints is %f" % (m, success_rate))

    return ave_rec_num, ave_rec_dis, success_rate


if __name__ == "__main__":
    with open("Data/SCA/Kyber768/es.txt", 'r') as g:
        solution = g.readlines()
    solution = np.array([int(x) for x in solution[0].split()])
    print("solution", solution)

    num_ine = []
    num_rec = []
    dis_rec = []
    suc_rat = []

    for m in tqdm(range(0, 2001, 40)):
        num_ine.append(m)
        print("\nThe number of sca approximate hints is", m)
        rec, dis, ratio = sol_approx_hints(m, 10, solution)
        num_rec.append(round(rec, 1))
        dis_rec.append(round(dis, 2))
        suc_rat.append(ratio)

    num_rec = [float(x) for x in num_rec]
    dis_rec = [float(x) for x in dis_rec]

    print("num_ine: ", num_ine)
    print("num_rec: ", num_rec)
    print("dis_rec: ", dis_rec)
    print("suc_rat: ", suc_rat)


