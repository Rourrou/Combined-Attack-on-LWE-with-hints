#!/usr/bin/python3
import random

# import solver
import numpy as np
import time
from tqdm import tqdm


# solving the secret only / secret error perfect hints of Kyber
def sol_perfect_hints(m, k, solution):
    ETA = 3

    nb_of_hints = 5000
    nb_of_unknowns = len(solution)
    # print("The number of unknowns", nb_of_unknowns)

    with open("DATA/LWE_with_hints/Perfect_hints/Simulate/secret_error/Kyber512/cbd_kyber512_v.txt", 'r') as f:
        lines_a = [next(f) for _ in range(nb_of_hints)]
    a = np.loadtxt(lines_a)

    with open("DATA/LWE_with_hints/Perfect_hints/Simulate/secret_error/Kyber512/cbd_kyber512_l.txt", 'r') as g:
        lines_b = [next(g) for _ in range(nb_of_hints)]
    b = np.loadtxt(lines_b)
    # print(b)

    if m == 0:
        E_int = [0] * nb_of_unknowns
        nb_correct = np.count_nonzero(solution == E_int)
        print("\nThe average recovered coefficients with %d hints is %d" % (m, nb_correct))
        short_vector = np.concatenate((np.array(E_int - solution), np.array([1])))
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
        a_selected = np.array([a[j] for j in indices])
        b_selected = np.array([b[j] for j in indices])
        #print("b_selected", b_selected)

        # 恢复全部私钥
        # s, n, d = solver.solve_perfect_hints_DGJ19(ETA, a_selected, b_selected, solution=solution, so_flag=None)
        # s, n, d = solver.solve_perfect_hints_DGJ19_CLT(ETA, a_selected, b_selected, solution=solution, so_flag=None)
        # s, n, d = solver.solve_perfect_hints_DGJ19_obo(ETA, a_selected, b_selected, solution=solution)
        s, n, d = solver.solve_perfect_hints_Del22(ETA, a_selected, b_selected, solution=solution, so_flag=None)
        # s, n, d = solver.solve_ineq_perfect_hints_Del22(ETA, a_selected, b_selected, solution=solution)

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

def sol_mod_q_hints(m, k, solution):
    ETA = 3

    nb_of_hints = 1000
    nb_of_unknowns = len(solution)
    print("The number of unknowns", nb_of_unknowns)

    with open("Data/Mod_q_hints/secret-only/Kyber128/v.txt", 'r') as f:
        lines_a = [next(f) for _ in range(nb_of_hints)]
    a = np.loadtxt(lines_a)

    with open("Data/Mod_q_hints/secret-only/Kyber128/l.txt", 'r') as g:
        lines_b = [next(g) for _ in range(nb_of_hints)]
    b = np.loadtxt(lines_b)
    # print(b)

    if m == 0:
        E_int = [0] * nb_of_unknowns
        nb_correct = np.count_nonzero(solution == E_int)
        print("\nThe average recovered coefficients with %d hints is %d" % (m, nb_correct))
        short_vector = np.concatenate((np.array(E_int - solution), np.array([1])))
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
        a_selected = np.array([a[j] for j in indices])
        b_selected = np.array([b[j] for j in indices])
        #print("b_selected", b_selected)



        # 恢复全部私钥
        # s, n, d = solver.solve_perfect_hints_DGJ19(ETA, a_selected, b_selected, solution=solution, so_flag=None)
        # s, n, d = solver.solve_perfect_hints_DGJ19_CLT(ETA, a_selected, b_selected, solution=solution, so_flag=None)
        # s, n, d = solver.solve_perfect_hints_DGJ19_obo(ETA, a_selected, b_selected, solution=solution)
        s, n, d = solver.solve_perfect_hints_Del22(ETA, a_selected, b_selected, solution=solution, so_flag=None)
        # s, n, d = solver.solve_ineq_perfect_hints_Del22(ETA, a_selected, b_selected, solution=solution)

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


def evaluate_inequalities_fast(a, b, solution):  # evaluate the direction of inequalities
    return (np.matmul(a, solution) - b) >= 0

# solving the translated secret only perfect hints of Kyber. Translate one perfect hint into 2 inequalities
def ineq_sol_perfect_hints(m, k, solution):
    ETA = 3

    nb_of_hints = 5000
    nb_of_unknowns = len(solution)
    print("nb_of_unknowns", nb_of_unknowns)

    with open("DATA/LWE_with_hints/Perfect_hints/Simulate/secret_error/Kyber128/cbd_kyber128_v.txt", 'r') as f:
        lines_a = [next(f) for _ in range(nb_of_hints)]
    a = np.loadtxt(lines_a)

    with open("DATA/LWE_with_hints/Perfect_hints/Simulate/secret_error/Kyber128/cbd_kyber128_l.txt", 'r') as g:
        lines_b = [next(g) for _ in range(nb_of_hints)]
    b = np.loadtxt(lines_b)

    if m == 0:
        E_int = [0] * nb_of_unknowns
        nb_correct = np.count_nonzero(solution == E_int)
        print("the average recovered coefficients with %d hints is %d" % (m, nb_correct))
        short_vector = np.concatenate((np.array(E_int - solution), np.array([1])))
        distance = np.linalg.norm(short_vector)
        distance = np.round(distance, 2)
        return nb_correct, distance, 0

    rec_num = []
    rec_dis = [] # the distance between recovered secret and solution
    num_correct = 0  # k次实验中，正确恢复完整私钥的次数

    # k次实验
    for i in range(k):
        # 选择m个索引
        indices = random.sample(range(nb_of_hints), m)
        a_selected = []
        b_selected = []
        for j in indices:
            a_selected.append(a[j])
            a_selected.append(a[j])
            b_selected.append(b[j] - 1)
            b_selected.append(b[j] + 1)

        a_selected = np.array(a_selected)
        b_selected = np.array(b_selected)
        print("a_selected: ", a_selected[:3], "\nb_selected:", b_selected[:5])
        is_geq_zero = evaluate_inequalities_fast(a_selected, b_selected, solution)
        print("is_geq_zero", is_geq_zero)

        # 恢复全部私钥
        s, n, d = solver.solve_ineq_perfect_hints_del22(ETA, a_selected, b_selected, is_geq_zero, max_nb_of_iterations=10, solution=solution)
        rec_num.append(n)
        rec_dis.append(d)
        if n == nb_of_unknowns:
            num_correct += 1
        print("Number of selected coeffs matches:{:d}/{:d}".format(n, nb_of_unknowns))

    ave_rec_num = np.round(np.mean(rec_num),2)
    ave_rec_dis = np.round(np.mean(rec_dis),2)
    success_rate = np.round(num_correct / k,2)

    print("the average recovered coefficients with %d ineqs is %f/%d" % (m, ave_rec_num, nb_of_unknowns))
    print("the success prob of recovering full ineqs with %d ineqs is %f" % (m, success_rate))

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
        print("\nThe number of equalities is", m)
        rec, dis, ratio = sol_mod_q_hints(m, 30, solution)
        num_rec.append(rec)
        dis_rec.append(dis)
        suc_rat.append(ratio)

    print("num_ine: ", num_ine)
    print("num_rec: ", [round(x,1) for x in num_rec])
    print("dis_rec: ", dis_rec)
    print("suc_rat: ", suc_rat)


