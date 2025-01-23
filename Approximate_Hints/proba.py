from math import factorial as fac
from math import log, ceil, erf, sqrt
import math
import numpy as np
from collections import defaultdict

def add(A, B):
    """
    compute the sum of two vectors
    """
    C = []
    for i in range(0, len(A)):
        C.append(A[i] + B[i])
    return C



def law_convolution(A, B):
    """ Construct the convolution of two laws (sum of independent variables from two input laws)
    :param A: first input law (dictionnary)
    :param B: second input law (dictionnary)
    """

    C = {}
    if (len(A) == 0):
        C = B
    elif (len(B) == 0):
        C = A
    for a in A:
        for b in B:
            c = a + b
            C[c] = C.get(c, 0) + A[a] * B[b]
    C = clean_dist(C)
    return C


def lim_law_convolution(A, B):
    """
    Construct the convolution of two laws (sum of independent variables from two input laws)
    Limit the number of iterms in C
    """
    C = {}
    lim = 256
    if len(A) == 0:
        C = B
    elif len(B) == 0:
        C = A
    for a in A:
        for b in B:
            c = a + b
            if abs(c) <= lim:
                C[c] = C.get(c, 0) + A[a] * B[b]
    C = clean_dist(C)
    return C


# SMY:20241216
def dis_CLT(A, c):
    """
    Construct the distribution of sc using the CLT, with known s dict and c
    Limit the number of iterms in D
    """
    D = {}
    lim = 2560
    if len(c) == 0:
        D = A
    E_s = sum(x*p for x,p in A.items()) # ��ֲ�������
    E_s2 = sum(x**2*p for x,p in A.items()) # E(s^2)
    Var_s = E_s2 - E_s**2 # ����

    E_cs = sum(c_i*E_s for c_i in c)
    #print("E_cs",E_cs)
    Var_cs = sum(c_i**2 * Var_s for c_i in c)
    #print("Var_cs", Var_cs)

    # CLT
    num_samples= 10000
    cs_samples = np.random.normal(E_cs, math.sqrt(Var_cs), num_samples)
    cs_dict = defaultdict(int)
    for sample in cs_samples:
        cs_dict[round(sample)] += 1

    total_samples = sum(cs_dict.values())
    for key in cs_dict:
        cs_dict[key] /= total_samples

    return cs_dict


# updating the distribution one by one minus the self, A = A-cB
def subtract_scale_dis(A, B, c):
    D = {}
    for a in A:
        for b in B:
            d = a - c*b
            D[d] = D.get(d, 0) + A[a] * B[b]
    D = clean_dist(D)
    return D

def lim_subtract_scale_dis(A, B, c):
    D = {}
    lim = 256
    for a, pa in A.items():
        for b, pb in B.items():
            d = a - c*b
            pro = D.get(d, 0) + pa * pb
            if abs(d)<=lim:
                D[d] = pro
    D = dict(sorted(D.items()))
    # D = clean_dist(D)
    return D

# updating the distribution one by one add the self, A = A+cB
def add_scale_dis(A, B, c):
    D = {}
    for a in A:
        for b in B:
            d = a + c*b
            D[d] = D.get(d, 0) + A[a] * B[b]
    D = clean_dist(D)
    return D

def lim_add_scale_dis(A, B, c):
    D = {}
    lim = 256
    for a, pa in A.items():
        for b, pb in B.items():
            d = a + c*b
            pro = D.get(d, 0) + pa * pb
            if abs(d) <= lim:
                D[d] = pro
    # D = clean_dist(D)
    return D


def law_product(A, B):
    """ Construct the law of the product of independent variables from two input laws
    :param A: first input law (dictionnary)
    :param B: second input law (dictionnary)
    """
    C = {}
    for a in A:
        for b in B:
            c = a * b
            C[c] = C.get(c, 0) + A[a] * B[b]  # C��Ϊһ��dict������ȡ����Ϊc��value�����޴˼�������0
    return C


def iter_law_convolution(A, i):
    """ compute the -ith forld convolution of a distribution (using double-and-add)
    :param A: first input law (dictionnary)
    :param i: (integer)
    """
    D = {0: 1.0}
    i_bin = bin(i)[2:]  # binary representation of n
    for ch in i_bin:
        #D = law_convolution(D, D)
        D = lim_law_convolution(D, D)
        D = clean_dist(D)
        if ch == '1':
            # D = law_convolution(D, A)
            D = lim_law_convolution(D, A)
            D = clean_dist(D)
    return D

def law_numproduct(A, i):
    """
    A.keys *= i, A.value *= 1
    :param A: (dictionnary)
    :param i: (integer)
    """
    B = {}
    for a in A:
        b = i * a
        B[b] = B.get(b, 0) + A[a]
    B = clean_dist(B)
    return B


def clean_dist(A):
    """ Clean a distribution to accelerate further computation (drop element of the support with proba less than 2^-300)
    :param A: input law (dictionnary)
    """
    B = {}
    for (x, y) in A.items():
        if y>2**(-300):
            B[x] = y
    return B


def tail_probability(D):
    """ Compute survival function of D """
    ma = max(D.keys())
    s = {ma + 1: 0.0}
    # Summing in reverse for better numerical precision (assuming tails are decreasing)
    for i in reversed(range(-ma, ma + 1)):
        s[i] = D.get(i, 0) + s[i + 1]
    return s


def min_mode(a):
    """
    return the min value of mode of array(a)
    """
    sta = {}
    for i in range(0, len(a)):
        b = a[i]
        sta[b] = sta.get(b, 0) + 1
    maxnum = 0
    for i in range(0, len(a)):
        b = a[i]
        if sta[b] > maxnum:
            maxnum = sta[b]
    mode = []
    for i in range(0, len(a)):
        b = a[i]
        if sta[b] == maxnum:
            mode.append(b)
    abs_mode = []
    for j in range(0, len(mode)):
        abs_mode.append(abs(mode[j]))
    ind = abs_mode.index(max(abs_mode))
    return mode[ind]