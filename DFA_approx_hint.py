import random
import solver
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14


def Plot_distribution():
    Value = []
    Sub = []
    solution = np.loadtxt('Data/DFA/es.txt', dtype=int)
    nb_of_hints = 5000
    q = int(3329/4)
    with open("Data/DFA/v.txt", 'r') as f:
        lines_v = [next(f) for _ in range(nb_of_hints)]
    V = np.loadtxt(lines_v)
    for i in range(len(V)):
        vs = np.dot(solution,V[i])
        Value.append(vs)
        Sub.append(q+10-vs)

    # Create histogram
    plt.hist(Value, bins=30, color='lightblue', edgecolor='black', alpha=0.7)
    plt.axvline(x=832, color='r', linestyle='--')
    plt.axvline(x=850, color='b', linestyle='--')
    plt.text(832, -38, 'q/4', color='red', ha='center', va='bottom', fontsize=14)

    # 设置标题和标签
    plt.xlabel("Value of $\langle s,v \\rangle$")
    plt.ylabel("Frequency")

    plt.show()



def sol_approx_hints(m, k, solution):
    ETA = 3
    Sigma = 3.3

    nb_of_hints = 1000
    nb_of_unknowns = len(solution)
    # print("The number of unknowns", nb_of_unknowns)

    with open("Data/DFA/v.txt", 'r') as f:
        lines_v = [next(f) for _ in range(nb_of_hints)]
    V = np.loadtxt(lines_v)

    with open("Data/DFA/l.txt", 'r') as g:
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
        s, n, d = solver.solve_approx_hints_Del22(ETA, Sigma, V_selected, L_selected, solution=solution, so_flag=None)
        rec_num.append(n)
        rec_dis.append(d)
        if n == nb_of_unknowns:
            num_correct += 1
        # print("Number of selected coeffs matches:{:d}/{:d}".format(n, nb_of_unknowns))

    ave_rec_num = np.mean(rec_num)
    ave_rec_dis = np.round(np.mean(rec_dis), 2)
    success_rate = num_correct / k

    print("The average recovered coefficients with %d approximate hints is %f/%d" % (m, ave_rec_num, nb_of_unknowns))
    print("The average recovered distances with %d approximate hints is %f/%d" % (m, ave_rec_dis, nb_of_unknowns))
    print("The success prob of recovering full coes with %d approximate hints is %f" % (m, success_rate))

    return ave_rec_num, ave_rec_dis, success_rate


def evaluate_inequalities_fast(v, l, solution):  # evaluate the direction of inequalities
    return (np.matmul(v, solution) - l) >= 0


if __name__ == "__main__":
    # nb_of_hints = 5000
    # q = 3329
    # solution = np.loadtxt('Data/DFA/es.txt', dtype=int)
    # print("solution", solution)
    #
    # num_ine = []
    # num_rec = []
    # dis_rec = []
    # suc_rat = []
    #
    # for m in tqdm(range(0, 601, 50)):
    #     num_ine.append(m)
    #     print("\nThe number of approximate hints is", m)
    #     rec, dis, ratio = sol_approx_hints(m, 10, solution)
    #     num_rec.append(round(rec, 1))
    #     dis_rec.append(round(dis, 2))
    #     suc_rat.append(ratio)
    #
    # num_rec = [float(x) for x in num_rec]
    # dis_rec = [float(x) for x in dis_rec]
    #
    # print("num_ine: ", num_ine)
    # print("num_rec: ", num_rec)
    # print("dis_rec: ", dis_rec)
    # print("suc_rat: ", suc_rat)

    Plot_distribution()


