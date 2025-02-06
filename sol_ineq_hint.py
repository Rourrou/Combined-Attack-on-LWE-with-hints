import random
import solver
import numpy as np
from tqdm import tqdm


def sol_ineq_hints(m, k, solution):
    ETA = 3

    nb_of_hints = 8000
    nb_of_unknowns = len(solution)
    print("nb_of_unknowns", nb_of_unknowns)

    with open("Data/Ineq Hints/secret error/Kyber512/v.txt", 'r') as f:
        lines_V = [next(f) for _ in range(nb_of_hints)]
    V = np.loadtxt(lines_V)

    with open("Data/Ineq Hints/secret error/Kyber512/l.txt", 'r') as g:
        lines_L = [next(g) for _ in range(nb_of_hints)]
    L = np.loadtxt(lines_L)

    if m == 0:
        E_int = [0] * nb_of_unknowns
        nb_correct = np.count_nonzero(solution == E_int)
        print("The average recovered coefficients with %d ineq hints is %d" % (m, nb_correct))
        short_vector = np.array(E_int - solution)
        distance = np.linalg.norm(short_vector)
        distance = np.round(distance, 2)
        print("The average distance with %d ineq hints is %d" % (m, distance))
        return nb_correct, distance, 0

    rec_num = []
    rec_dis = [] # the distance between recovered secret and solution
    num_correct = 0  # k次实验中，正确恢复完整私钥的次数

    # k次实验
    for i in range(k):
        # 选择m个索引
        indices = random.sample(range(nb_of_hints), m)
        V_selected = np.array([V[j] for j in indices])
        L_selected = np.array([L[j] for j in indices])
        is_geq_zero = evaluate_inequalities_fast(V_selected, L_selected, solution)
        # print(is_geq_zero)

        # 恢复全部私钥
        s, n, d = solver.solve_ineq_hints_del22(ETA, V_selected, L_selected, is_geq_zero, solution=solution)
        rec_num.append(n)
        rec_dis.append(d)
        if n == nb_of_unknowns:
            num_correct += 1
        # print("Number of selected coeffs matches:{:d}/{:d}".format(n, nb_of_unknowns))

    ave_rec_num = np.round(np.mean(rec_num), 2)
    ave_rec_dis = np.round(np.mean(rec_dis), 2)
    success_rate = np.round(num_correct / k, 2)

    print("The average recovered coefficients with %d ineq hints is %f/%d" % (m, ave_rec_num, nb_of_unknowns))
    print("The average recovered distances with %d ineq hints is %f/%d" % (m, ave_rec_dis, nb_of_unknowns))
    print("The success prob of recovering full coes with %d ineq hints is %f" % (m, success_rate))

    return ave_rec_num, ave_rec_dis, success_rate


def evaluate_inequalities_fast(v, l, solution):  # evaluate the direction of inequalities
    return (np.matmul(v, solution) - l) >= 0


if __name__ == "__main__":
    with open("Data/Ineq Hints/secret error/Kyber512/es.txt", 'r') as g:
        solution = g.readlines()
    solution = np.array([int(x) for x in solution[0].split()])
    print("solution", solution)

    num_ine = []
    num_rec = []
    dis_rec = []
    suc_rat = []

    for m in tqdm(range(0, 6001, 200)):
        num_ine.append(m)
        print("\nThe number of approximate hints is", m)
        rec, dis, ratio = sol_ineq_hints(m, 20, solution)
        num_rec.append(round(rec, 1))
        dis_rec.append(round(dis, 2))
        suc_rat.append(ratio)

    num_rec = [float(x) for x in num_rec]
    dis_rec = [float(x) for x in dis_rec]
    suc_rat = [float(x) for x in suc_rat]

    print("num_ine: ", num_ine)
    print("num_rec: ", num_rec)
    print("dis_rec: ", dis_rec)
    print("suc_rat: ", suc_rat)


