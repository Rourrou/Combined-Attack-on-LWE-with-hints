import random
import solver
import numpy as np
from tqdm import tqdm

# Data/Ineq Hints/secret error/LWE_80_10_3_2/
# Data/ShaoYan/LWE_80_10_3_2/T9
def sol_ineq_hints(m, solution):
    ETA = 10
    nb_of_hints = 1001
    nb_of_unknowns = len(solution)
    print("nb_of_unknowns", nb_of_unknowns)

    with open("Data/ShaoYan/LWE_80_10_3_2/T9/V.txt", 'r') as f:
        lines_V = [next(f) for _ in range(nb_of_hints)]
    V = np.loadtxt(lines_V)

    with open("Data/ShaoYan/LWE_80_10_3_2/T9/l_ori.txt", 'r') as g:
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
        return short_vector, nb_correct, distance


    num_correct = 0  # k次实验中，正确恢复完整私钥的次数

    V_selected = np.array(V[:m,:])
    # print("V_selected",V_selected)
    L_selected = np.array(L[:m])
    LP = V_selected @ solution

    is_geq_zero = evaluate_inequalities_fast(V_selected, L_selected, solution)
    # print(is_geq_zero)

    # 恢复全部私钥
    s, n, d = solver.solve_ineq_hints_del22(ETA, V_selected, L_selected, is_geq_zero, solution=solution)

    s_str = " ".join(map(str, s))
    with open("Data/ShaoYan/LWE_80_10_3_2/T9/es_new.txt", "a") as f:
        _ = f.write(s_str+ "\n")
    print(f"Vector s has been saved")


    print("The recovered coefficients with %d ineq hints is %f/%d" % (m, n, nb_of_unknowns))
    print("The recovered distances with %d ineq hints is %f/%d" % (m, d, nb_of_unknowns))

    return s, n, d


def evaluate_inequalities_fast(v, l, solution):  # evaluate the direction of inequalities
    return (np.matmul(v, solution) - l) >= 0


if __name__ == "__main__":
    with open("Data/ShaoYan/LWE_80_10_3_2/T9/es.txt", 'r') as g:
        solution = g.readlines()
    solution = np.array([int(x) for x in solution[0].split()])
    print("solution", solution)

    num_ine = []
    num_rec = []
    dis_rec = []
    suc_rat = []

    for m in tqdm(range(0, 1001, 100)):
        num_ine.append(m)
        print("\nThe number of approximate hints is", m)
        s_new, rec, dis = sol_ineq_hints(m, solution)
        num_rec.append(round(rec, 1))
        dis_rec.append(round(dis, 2))

    num_rec = [float(x) for x in num_rec]
    dis_rec = [float(x) for x in dis_rec]

    print("num_ine: ", num_ine)
    print("num_rec: ", num_rec)
    print("dis_rec: ", dis_rec)


