import random
import solver
import numpy as np
from tqdm import tqdm


# solving the secret error perfect hints of Kyber
def sol_perfect_hints(m, k, solution):
    ETA = 40
    nb_of_hints = 1000
    nb_of_unknowns = len(solution)
    # print("The number of unknowns", nb_of_unknowns)

    with open("Data/Perfect Hints/secret error/LWE_80/v.txt", 'r') as f:
        lines_V = [next(f) for _ in range(nb_of_hints)]
    V = np.loadtxt(lines_V)

    with open("Data/Perfect Hints/secret error/LWE_80/l.txt", 'r') as g:
        lines_L = [next(g) for _ in range(nb_of_hints)]
    L = np.loadtxt(lines_L)
    # print(b)

    if m == 0:
        E_int = [0] * nb_of_unknowns
        nb_correct = np.count_nonzero(solution == E_int)
        print("\nThe average recovered coefficients with %d perfect hints is %d" % (m, nb_correct))
        short_vector = np.array(E_int - solution)
        distance = np.linalg.norm(short_vector)
        distance = np.round(distance, 2)
        print("The average distance with %d ineq hints is %d" % (m, distance))
        return nb_correct, distance, 0

    rec_num = []
    rec_dis = []  # the distance between recovered secret and solution
    num_correct = 0  # k次实验中，正确恢复完整私钥的次数

    # k次实验
    for i in range(k):
        # 选择m个索引
        indices = random.sample(range(nb_of_hints), m)
        V_selected = np.array([V[j] for j in indices])
        L_selected = np.array([L[j] for j in indices])
        #print("b_selected", b_selected)

        # 恢复全部私钥
        s, n, d = solver.solve_perfect_hints_Del22(ETA, V_selected, L_selected, solution=solution)

        rec_num.append(n)
        rec_dis.append(d)
        if n == nb_of_unknowns:
            num_correct += 1
        print("Number of selected coeffs matches:{:d}/{:d}".format(n, nb_of_unknowns))

    ave_rec_num = np.mean(rec_num)
    ave_rec_dis = np.round(np.mean(rec_dis), 2)
    success_rate = num_correct / k

    print("the average recovered coefficients with %d perfect hints is %f/%d" % (m, ave_rec_num, nb_of_unknowns))
    print("The average recovered distances with %d perfect hints is %f/%d" % (m, ave_rec_dis, nb_of_unknowns))
    print("the success prob of recovering full coes with %d perfect hints is %f" % (m, success_rate))

    return ave_rec_num, ave_rec_dis, success_rate


if __name__ == "__main__":
    with open("Data/Perfect Hints/secret error/LWE_80/es.txt", 'r') as g:
        solution = g.readlines()
    solution = np.array([int(x) for x in solution[0].split()])
    print("solution", solution)

    num_hint = []
    num_rec = []
    dis_rec = []
    suc_rat = []

    for m in tqdm(range(20, 401, 20)):
        num_hint.append(m)
        print("\nThe number of perfect hints is", m)
        rec, dis, ratio = sol_perfect_hints(m, 50, solution)
        num_rec.append(round(rec, 1))
        dis_rec.append(round(dis, 2))
        suc_rat.append(ratio)

    num_rec = [float(x) for x in num_rec]
    dis_rec = [float(x) for x in dis_rec]
    suc_rat = [float(x) for x in suc_rat]

    print("num_ine: ", num_hint)
    print("num_rec: ", num_rec)
    print("dis_rec: ", dis_rec)
    print("suc_rat: ", suc_rat)

