import numpy as np


def inequalities(m):
    # 读取txt文件内容
    with open("Data/Ineq Hints/secret error/Kyber512/v.txt", "r") as txt_file:
        lines = txt_file.readlines()[:m]

    # 创建.py文件并写入内容
    with open("Data/Ineq Hints/secret error/Kyber512/inequalities.py", "w") as py_file:
        py_file.write(f"coeffs = [\n")
        for line in lines:
            # 移除行尾的换行符，并按空格分割元素
            elements = line.strip().split(" ")
            # 将元素组合成字符串，用逗号加空格隔开
            vector_str = ", ".join(elements)
            # 写入到.py文件中
            py_file.write(f"[{vector_str}],\n")
        py_file.write(f"]\n")


def signs(m):
    # 创建.py文件并写入内容
    with open("Data/Ineq Hints/secret error/Kyber512/inequalities.py", "a") as py_file:
        py_file.write(f"signs = [\n")
        for i in range(m):
            py_file.write(f'"<=",\n')
        py_file.write(f"]\n")


def bs(m):
    # 读取txt文件内容
    with open("Data/Ineq Hints/secret error/Kyber512/l.txt", "r") as txt_file:
        lines = [line.strip() for line in txt_file.readlines()[:m]]

    # 创建.py文件并写入内容
    with open("Data/Ineq Hints/secret error/Kyber512/inequalities.py", "a") as py_file:
        py_file.write(f"bs = [\n")
        for line in lines:
            py_file.write(f"{line},\n")
        py_file.write(f"]\n")


def is_corrects(m):
    with open("Data/Ineq Hints/secret error/Kyber512/inequalities.py", "a") as py_file:
        py_file.write(f"is_corrects = [\n")
        for i in range(m):
            py_file.write(f"True,\n")
        py_file.write(f"]\n")


def p_corrects(m):
    with open("Data/Ineq Hints/secret error/Kyber512/inequalities.py", "a") as py_file:
        py_file.write(f"p_corrects = [\n")
        for i in range(m):
            py_file.write(f"1,\n")
        py_file.write(f"]\n")


def run_data(m, n):
    # 读取txt文件内容
    with open("Data/Ineq Hints/secret error/Kyber512/es.txt", "r") as txt_file:
        elements = txt_file.read().split()

    key_e = elements[:n]
    key_s = elements[n:]

    # 创建.py文件并写入内容
    with open("Data/Ineq Hints/secret error/Kyber512/run_data.py", "w") as py_file:
        py_file.write("key_e = [" + ", ".join(key_e) + "]\n")
        py_file.write("key_s = [" + ", ".join(key_s) + "]\n")

        py_file.write(f"key = key_e + key_s\n")
        py_file.write(f"max_delta_v = None\n")
        py_file.write(f"filtered_cts = {m}\n")
        py_file.write(f"ineqs = {m}\n")
        py_file.write(f"correct_ineqs = {m}\n")
        py_file.write(f"recovered_coefficients = {m}\n")


def lwe_instance(n):
    # 读取公钥A
    with open("Data/Ineq Hints/secret error/Kyber512/lwe_instance.py", "w") as py_file:
        py_file.write(f"a = [")
        py_file.write(f"]\n")

    # 读取LWE值b
    with open("Data/Ineq Hints/secret error/Kyber512/lwe_instance.py", "a") as py_file:
        py_file.write(f"b = [")
        py_file.write(f"]\n")

    # 读取私钥e+s
    with open("Data/Ineq Hints/secret error/Kyber512/es.txt", "r") as txt_file:
        elements = txt_file.read().split()

    key_e = elements[:n]
    key_s = elements[n:]

    # 创建.py文件并写入内容
    with open("Data/Ineq Hints/secret error/Kyber512/lwe_instance.py", "a") as py_file:
        py_file.write("e = [" + ", ".join(key_e) + "]\n")
        py_file.write("s = [" + ", ".join(key_s) + "]\n")
        py_file.write(f"key = e + s\n")


# 将Del22生成的数据转为HMS23的格式
def ineq2ineq_HMS23(m,n):
    inequalities(m)
    signs(m)
    bs(m)
    is_corrects(m)
    p_corrects(m)
    run_data(m, n)
    lwe_instance(n)


if __name__ == '__main__':
    m = 8000
    n = 512
    ineq2ineq_HMS23(m, n)
