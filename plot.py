import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16

Kyber111_approx_num = []
Kyber111_approx_emb = []
Kyber111_approx_dis = []
Kyber111_approx_ps = []
Kyber111_approx_com = []

LWE80_approx_num = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
LWE80_approx_emb_est = [59.06, 52.81, 46.53, 40.25, 33.82, 26.49, 21.14, 15.18, 11.11, 8.29]
LWE80_approx_emb_pra = [62.00, 53.00, 53.00, 52.00, 41.00, 37.00, 35.00, 32.00, 28.00, 23.00]
LWE80_approx_dis = [53.58, 52.78, 51.92, 51.02, 50.19, 49.05, 48.41, 47, 46.58, 45.34]
LWE80_approx_ps = [59.06, 58.19, 57.24, 56.23, 55.29, 53.97, 53.22, 51.54, 51.03, 49.49]
LWE80_approx_com_est = [58.19, 51.00, 43.76, 36.37, 28.13, 21.44, 14.23, 9.79, 6.63, 4.10]
LWE80_approx_com_est = [62.00, 49.60, 49.80, 40.80, 34.40, 33.00, 28.60, 27.80, 23.40, 19.80]

Kyber128_approx_num = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 580, 590, 600, 610, 620, 630, 640, 650]
Kyber128_approx_emb = [50.58, 46.37, 42.19, 37.96, 33.7, 29.26, 24.43, 22.05, 19.55, 16.5, 14.13, 12.48, 10.71, 9.46, 8.56, 7.68, 6.72, 5.64, 4.82, 4.44, 4.06]
Kyber128_approx_dis = [20.42, 20.36, 20.1, 19.76, 19.44, 19, 18.61, 18.1, 17.86, 17.42, 17.03, 16.52, 16.19, 15.79, 15.51, 14.97, 14.64, 14.13, 13.95, 13.44, 13.09, 12.89, 12.47, 12.04, 11.69, 11.48, 11.19, 10.82, 10.48, 10.42, 9.98, 9.83, 9.46, 9.31, 9.05, 8.76, 8.51, 8.18, 8.03, 7.63, 7.32, 7.15, 6.58, 6.33, 6.08, 5.37, 4.91, 4.41, 3.6, 2.97, 2.66, 1.84, 1.8, 1.08, 1.15, 0.79, 0.58, 0.37, 0.11, 0.1, 0.14, 0.06, 0.04, 0.03, 0.01, 0.0]
Kyber128_approx_ps = [50.58, 50.45, 49.89, 49.14, 48.43, 47.43, 46.52, 45.31, 44.72, 43.62, 42.63, 41.29, 40.40, 39.29, 38.49, 36.91, 35.89, 34.27, 33.68, 31.95, 30.69, 29.95, 28.31, 26.13, 24.66, 24.19, 23.53, 22.64, 21.80, 21.65, 20.49, 20.08, 18.85, 18.31, 17.34, 16.18, 15.08, 14.31, 13.97, 13.00, 12.19, 11.71, 9.97, 9.56, 9.14, 7.77, 6.69, 5.15, 4.00, 2.76, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00]
Kyber128_approx_com = [50.58, 46.21, 41.44, 36.48, 31.37, 24.96, 22.00, 18.57, 14.93, 12.78, 10.48, 9.00, 7.83, 6.51, 5.07, 4.41, 3.87, 3.18, 2.46, 2.00, 2.00]

Kyber128_ineq_num = [0, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680, 720, 760, 800]
Kyber128_ineq_emb = [49.21, 47.70, 45.79, 44.42, 43.52, 42.97, 42.01, 40.99, 40.16, 39.43, 38.67, 38.10, 37.76, 37.42, 37.06, 36.73, 36.51, 36.29, 35.90, 36.55, 36.39]
Kyber128_ineq_dis = [18.52, 18.52, 18.52, 18.52, 17.89, 17.09, 16.82, 15.94, 15.07, 14.03, 13.35, 12, 11.58, 10.34, 9.8, 6.56, 4.24, 3.61, 0, 0, 0]
Kyber128_ineq_ps = [49.29, 49.29, 49.29, 49.29, 47.78, 45.78, 45.08, 42.72, 40.26, 37.08, 35.09, 29.91, 28.19, 22.72, 21.32, 10.66, 4.76, 3.23, 2, 2, 2]
Kyber128_ineq_com = [49.21, 47.19, 45.60, 43.47, 40.92, 38.07, 34.82, 31.29, 26.47, 25.87, 20.60, 19.08, 14.93, 13.67, 7.50, 3.58, 2.24, 2.00, 2.00, 2, 2]

Kyber128_perfect_num = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
Kyber128_perfect_emb = [50.58, 41.76, 32.17, 22.52, 14.84, 9.72, 6.28, 3.64, 2]
Kyber128_perfect_dis = [20.02, 20.01, 19.98, 19.9, 19.66, 19.59, 19.31, 19.14, 18.97, 18.79, 18.65, 16.51, 14.28, 12.37, 10.95, 9.4, 7.75, 4.07, 1.16, 0]
Kyber128_perfect_ps = [50.58, 50.56, 50.49, 50.32, 49.79, 49.63, 49.00, 48.61, 48.23, 47.81, 47.48, 42.14, 35.66, 28.90, 23.48, 19.29, 13.67, 4.74, 2.00, 2.00]
Kyber128_perfect_com = [50.58, 41.67, 31.92, 22.04, 14.41, 9.27, 5.41, 3.20, 2]


Kyber128_modular_num = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
Kyber128_modular_emb = [50.58, 43.67, 36.67, 29.29, 22.74, 18.01, 13.02, 9.23, 6.13, 3.57, 2, 2, 2,]
Kyber128_modular_dis = []
Kyber128_modular_ps = [50.58, 50.58, 50.58, 50.58, 50.58, 50.58, 50.58, 50.58, 50.58, 50.58, 50.58, 50.58, 50.58]
Kyber128_modular_com = []


Kyber256_approx_num = [0, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680, 720, 760, 800, 840, 880, 920, 960, 1000]
Kyber256_approx_emb = [165.1, 151.78, 139.88, 129.01, 119.73, 110.94, 103.4, 96.37, 90.04, 84.16, 78.7, 73.68, 69.16, 65.43, 62.44, 59.99, 57.96, 56.16, 54.58, 53.14, 51.82, 50.52, 49.38, 48.29, 47.28, 46.34]
Kyber256_approx_dis = [26.91, 26.39, 25.45, 24.6, 23.65, 22.08, 21.07, 20.21, 18.78, 18.23, 16.86, 16.03, 15.29, 14.36, 13.62, 13.1, 11.79, 11.05, 10.44, 9.17, 6.05, 4.61, 1.08, 0.97, 0, 0]
Kyber256_approx_ps = [165.10, 163.86, 161.58, 159.48, 157.08, 153.01, 150.31, 147.96, 143.92, 142.32, 138.22, 135.64, 133.28, 130.22, 127.70, 125.88, 121.10, 118.26, 115.84, 110.50, 95.01, 86.03, 46.24, 43.35, 2.00, 2.00]
Kyber256_approx_com = [165.10, 150.66, 137.00, 125.08, 114.22, 103.46, 94.62, 86.83, 78.84, 72.46, 65.09, 58.81, 53.10, 47.40, 42.52, 38.58, 32.97, 28.73, 24.51, 21.72, 13.95, 10.49, 3.26, 2.73, 2.00, 2.00]

Kyber256_ineq_num = []
Kyber256_ineq_emb = []
Kyber256_ineq_dis = []
Kyber256_ineq_ps = []
Kyber256_ineq_com = []

Kyber256_perfect_num = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100]
Kyber256_perfect_emb = [165.1, 151.71, 138.93, 126.74, 115.08, 103.92, 93.23, 82.95, 72.99, 63.2, 53.22, 42.28, 30.14, 20.46, 13.11, 8.69, 5.05, 3.23, 2, 2, 2]
Kyber256_perfect_dis = [27.35, 27.33, 27.3, 27.16, 27.02, 26.73, 26.57, 26.32, 25.95, 25.87, 25.5, 25.15, 25.03, 24.75, 24.43, 24.34, 24.02, 23.58, 23.41, 23.01, 22.84, 20.1, 17.2, 15.37, 13.71, 12.23, 9.43, 3.69, 0.11, 0]
Kyber256_perfect_ps = [165.10, 165.05, 164.98, 164.66, 164.33, 163.64, 163.26, 162.67, 161.78, 161.59, 160.69, 159.84, 159.54, 158.86, 158.06, 157.84, 157.04, 155.93, 155.50, 154.48, 154.04, 146.8, 138.4, 132.7, 127.2, 122.0, 111.0, 78.74, 8.67, 2.00]
Kyber256_perfect_com = [165.10, 151.66, 138.83, 126.37, 114.51, 102.98, 92.15, 80.64, 70.34, 60.53, 50.00, 38.37, 24.87, 17.50, 11.15, 7.53, 4.41, 2.29, 2, 2, 2]


Kyber512_dfa_num = [0, 2, 4, 6, 8, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170]
Kyber512_dfa_emb = [405.53, 404.88, 404.23, 403.59, 402.96, 402.32, 399.19, 396.11, 393.08, 390.12, 387.19, 384.27, 381.41, 378.58, 375.8, 373.05, 370.35, 367.66, 365.03, 362.42, 359.85, 357.3]
Kyber512_dfa_dis = [40.09, 32.48, 28.04, 25.10, 22.78, 21.31, 17.28, 14.58, 12.54, 11.38, 10.01, 9.06, 8.14, 7.25, 6.28, 5.37, 4.22, 3.27, 2.42, 1.56, 0.77, 0.23]
Kyber512_dfa_ps = [405.53, 380.68, 364.64, 353.19, 343.60, 337.22, 318.25, 303.99, 292.09, 284.78, 275.51, 268.60, 261.45, 254.01, 245.20, 236.07, 222.91, 210.05, 196.14, 177.97, 153.08, 119.67]
Kyber512_dfa_com = [405.53, 380.09, 363.52, 351.57, 341.49, 334.67, 313.68, 297.65, 284.2, 275.36, 264.82, 256.64, 248.38, 239.98, 230.43, 220.73, 207.51, 194.72, 181.13, 163.84, 140.62, 109.71]


def plot_LWE80_approx_hint():
    LWE80_approx_num = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
    LWE80_approx_emb_est = [59.06, 52.81, 46.53, 40.25, 33.82, 26.49, 21.14, 15.18, 11.11, 8.29]
    LWE80_approx_emb_pra = [62.00, 53.00, 53.00, 52.00, 41.00, 37.00, 35.00, 32.00, 28.00, 23.00]
    LWE80_approx_dis = [53.58, 52.78, 51.92, 51.02, 50.19, 49.05, 48.41, 47, 46.58, 45.34]
    LWE80_approx_ps = [59.06, 58.19, 57.24, 56.23, 55.29, 53.97, 53.22, 51.54, 51.03, 49.49]
    LWE80_approx_com_est = [59.06, 51.00, 43.76, 36.37, 28.13, 21.44, 14.23, 9.79, 6.63, 4.10]
    LWE80_approx_com_pra = [62.00, 49.60, 49.80, 40.80, 34.40, 33.00, 28.60, 27.80, 23.40, 19.80]

    # 只取横坐标在800及以下的值
    LWE80_approx_num = np.array(LWE80_approx_num)
    mask = LWE80_approx_num <= 45

    # 画图
    plt.figure(figsize=(8, 6))
    plt.plot(LWE80_approx_num[mask], np.array(LWE80_approx_emb_est)[mask], 'v', markersize=5, markerfacecolor='none', linestyle='dashed',
             linewidth=1, label='Prediction (DDGR20)', color='black')
    plt.plot(LWE80_approx_num[mask], np.array(LWE80_approx_com_est)[mask], 'o', markersize=5, markerfacecolor='none', linestyle='dashed',
             linewidth=1, label='Prediction (Our)', color='black')
    plt.plot(LWE80_approx_num[mask], np.array(LWE80_approx_emb_pra)[mask], 's', markersize=5, markerfacecolor='none', linestyle='dashed',
             linewidth=1, label='Experiment (DDGR20)', color='blue')
    plt.plot(LWE80_approx_num[mask], np.array(LWE80_approx_com_pra)[mask], '*', markersize=5, markerfacecolor='none',
             linestyle='dashed',
             linewidth=1, label='Experiment (Our)', color='red')

    # Labels and title
    plt.xlabel("Number of hints")
    plt.ylabel("BKZ-$\\beta$")
    plt.legend(loc='upper right', fontsize=14)

    plt.show()

def plot_Kyber128_approx_hint_LR_PS():
    Kyber128_approx_num = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200,
                           210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380,
                           390, 400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560,
                           570, 580, 590, 600, 610, 620, 630, 640, 650]
    Kyber128_approx_emb = [50.58, 46.37, 42.19, 37.96, 33.7, 29.26, 24.43, 22.05, 19.55, 16.5, 14.13, 12.48, 10.71,
                           9.46, 8.56, 7.68, 6.72, 5.64, 4.82, 4.44, 4.06, 3.61, 3.15, 2.5, 2, 2]

    Kyber128_approx_ps = [50.58, 50.45, 49.89, 49.14, 48.43, 47.43, 46.52, 45.31, 44.72, 43.62, 42.63, 41.29, 40.40,
                          39.29, 38.49, 36.91, 35.89, 34.27, 33.68, 31.95, 30.69, 29.95, 28.31, 26.13, 24.66, 24.19,
                          23.53, 22.64, 21.80, 21.65, 20.49, 20.08, 18.85, 18.31, 17.34, 16.18, 15.08, 14.31, 13.97,
                          13.00, 12.19, 11.71, 9.97, 9.56, 9.14, 7.77, 6.69, 5.15, 4.00, 2.76, 2.00, 2.00, 2.00, 2.00,
                          2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00]


    # 只取横坐标在200及以下的值
    Kyber128_approx_num = np.array(Kyber128_approx_num)
    indices_emb = np.max(np.nonzero(Kyber128_approx_num <= 200)[0])
    indices_ps = np.max(np.nonzero(Kyber128_approx_num <= 500)[0])

    # 画图
    plt.figure(figsize=(8, 6))
    plt.plot(Kyber128_approx_num[:indices_emb], np.array(Kyber128_approx_emb)[:indices_emb], 'v', markersize=5, markerfacecolor='none', linestyle='dashed',
             linewidth=1, label='Lattice Reduction', color='blue')
    plt.plot(Kyber128_approx_num[:indices_ps], np.array(Kyber128_approx_ps)[:indices_ps], 'o', markersize=5, markerfacecolor='none', linestyle='dashed',
             linewidth=1, label='Probabilistic Statistic', color='green')

    # Labels and title
    plt.xlabel("Kyber128 Approx Num")
    plt.ylabel("Values")
    plt.title("Comparison of Kyber128 Approximation Hints")
    plt.legend()

    plt.show()


def plot_Kyber128_ineq_hint_LR_PS():
    Kyber128_ineq_num = [0, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680, 720, 760,
                         800]
    Kyber128_ineq_emb = [49.21, 47.70, 45.79, 44.42, 43.52, 42.97, 42.01, 40.99, 40.16, 39.43, 38.67, 38.10, 37.76, 37.42, 37.06, 36.73, 36.51, 36.29, 35.90, 36.55, 36.39]
    Kyber128_ineq_ps = [49.29, 49.29, 49.29, 49.29, 47.78, 45.78, 45.08, 42.72, 40.26, 37.08, 35.09, 29.91, 28.19,
                        22.72, 21.32, 10.66, 4.76, 3.23, 2, 2, 2]

    # 只取横坐标在200及以下的值
    Kyber128_ineq_num = np.array(Kyber128_ineq_num)
    indices = np.max(np.nonzero(Kyber128_ineq_num <= 760)[0])
    print("indices", indices)

    # 画图
    plt.figure(figsize=(8, 6))
    plt.plot(Kyber128_ineq_num[:indices], np.array(Kyber128_ineq_emb)[:indices], 'v', markersize=5, markerfacecolor='none', linestyle='dashed',
             linewidth=1, label='Lattice Reduction [DDGR20]', color='blue')
    plt.plot(Kyber128_ineq_num[:indices], np.array(Kyber128_ineq_ps)[:indices], 'o', markersize=5, markerfacecolor='none', linestyle='dashed',
             linewidth=1, label='Probabilistic Statistic', color='green')

    # Labels and title
    plt.xlabel("Kyber128 ineq Num")
    plt.ylabel("Values")
    plt.title("Comparison of Kyber128 ineq Hints")
    plt.legend()

    plt.show()


def plot_Kyber128_perfect_hint_LR_PS():
    Kyber128_perfect_num = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    Kyber128_perfect_emb = [50.58, 41.76, 32.17, 22.52, 14.84, 9.72, 6.28, 3.64, 2, 2, 2, 2, 2, 2]
    Kyber128_perfect_ps = [50.58, 50.56, 50.49, 50.32, 49.79, 49.63, 49.00, 48.61, 48.23, 47.81, 47.48, 42.14, 35.66,
                           28.90, 23.48, 19.29, 13.67, 4.74, 2.00, 2.00]

    # 只取横坐标在200及以下的值
    Kyber128_perfect_num = np.array(Kyber128_perfect_num)
    indices_LR = np.max(np.nonzero(Kyber128_perfect_num <= 40)[0])
    indices_PS = np.max(np.nonzero(Kyber128_perfect_num <= 200)[0])


    # 画图
    plt.figure(figsize=(8, 6))
    plt.plot(Kyber128_perfect_num[:indices_LR], np.array(Kyber128_perfect_emb)[:indices_LR], 'v', markersize=5, markerfacecolor='none', linestyle='dashed',
             linewidth=1, label='Lattice Reduction [DDGR20]', color='blue')
    plt.plot(Kyber128_perfect_num[:indices_PS], np.array(Kyber128_perfect_ps)[:indices_PS], 'o', markersize=5, markerfacecolor='none', linestyle='dashed',
             linewidth=1, label='Probabilistic Statistic', color='green')

    # Labels and title
    plt.xlabel("Kyber128 perfect Num")
    plt.ylabel("Values")
    plt.title("Comparison of Kyber128 perfect Hints")
    plt.legend()

    plt.show()


def plot_Kyber128_approx_hint_Com():
    Kyber128_approx_num = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200,
                           210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380,
                           390, 400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560,
                           570, 580, 590, 600, 610, 620, 630, 640, 650]
    Kyber128_approx_emb = [50.58, 46.37, 42.19, 37.96, 33.7, 29.26, 24.43, 22.05, 19.55, 16.5, 14.13, 12.48, 10.71,
                           9.46, 8.56, 7.68, 6.72, 5.64, 4.82, 4.44, 4.06, 3.61, 3.15, 2.5, 2, 2]

    Kyber128_approx_ps = [50.58, 50.45, 49.89, 49.14, 48.43, 47.43, 46.52, 45.31, 44.72, 43.62, 42.63, 41.29, 40.40,
                          39.29, 38.49, 36.91, 35.89, 34.27, 33.68, 31.95, 30.69, 29.95, 28.31, 26.13, 24.66, 24.19,
                          23.53, 22.64, 21.80, 21.65, 20.49, 20.08, 18.85, 18.31, 17.34, 16.18, 15.08, 14.31, 13.97,
                          13.00, 12.19, 11.71, 9.97, 9.56, 9.14, 7.77, 6.69, 5.15, 4.00, 2.76, 2.00, 2.00, 2.00, 2.00,
                          2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00]
    Kyber128_approx_com = [50.58, 46.21, 41.44, 36.48, 31.37, 24.96, 22.00, 18.57, 14.93, 12.78, 10.48, 9.00, 7.83,
                           6.51, 5.07, 4.41, 3.87, 3.18, 2.46, 2.00, 2.00]

    # 只取横坐标在200及以下的值
    Kyber128_approx_num = np.array(Kyber128_approx_num)
    indices = np.max(np.nonzero(Kyber128_approx_num <= 140)[0])

    # 画图
    plt.figure(figsize=(8, 6))
    plt.plot(Kyber128_approx_num[:indices], np.array(Kyber128_approx_emb)[:indices], 'v', markersize=5, markerfacecolor='none', linestyle='dashed',
             linewidth=1, label='Lattice Reduction', color='blue')
    plt.plot(Kyber128_approx_num[:indices], np.array(Kyber128_approx_ps)[:indices], 'o', markersize=5, markerfacecolor='none', linestyle='dashed',
             linewidth=1, label='Probabilistic Statistic', color='green')
    plt.plot(Kyber128_approx_num[:indices], np.array(Kyber128_approx_com)[:indices], 's', markersize=5,
             markerfacecolor='none', linestyle='dashed',
             linewidth=1, label='Combinatorial Attack', color='red')

    # Labels and title
    plt.xlabel("Number of approximate hints")
    plt.ylabel("BKZ-$\\beta$")
    plt.legend(loc='upper right')

    plt.show()


def plot_Kyber128_ineq_hint_Com():
    Kyber128_ineq_num = [0, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680, 720, 760,
                         800]
    Kyber128_ineq_emb = [49.21, 47.70, 45.79, 44.42, 43.52, 42.97, 42.01, 40.99, 40.16, 39.43, 38.67, 38.10, 37.76, 37.42, 37.06, 36.73, 36.51, 36.29, 35.90, 36.55, 36.39]
    Kyber128_ineq_ps = [49.29, 49.29, 49.29, 49.29, 47.78, 45.78, 45.08, 42.72, 40.26, 37.08, 35.09, 29.91, 28.19,
                        22.72, 21.32, 10.66, 4.76, 3.23, 2, 2, 2]
    Kyber128_ineq_com = [49.21, 47.19, 45.60, 43.47, 40.92, 38.07, 34.82, 31.29, 26.47, 25.87, 20.60, 19.08, 14.93,
                         13.67, 7.50, 3.58, 2.24, 2.00, 2.00, 2, 2]

    # 只取横坐标在200及以下的值
    Kyber128_ineq_num = np.array(Kyber128_ineq_num)
    indices = np.max(np.nonzero(Kyber128_ineq_num <= 600)[0])
    print("indices", indices)

    # 画图
    plt.figure(figsize=(8, 6))
    plt.plot(Kyber128_ineq_num[:indices], np.array(Kyber128_ineq_emb)[:indices], 'v', markersize=5, markerfacecolor='none', linestyle='dashed',
             linewidth=1, label='Lattice Reduction', color='blue')
    plt.plot(Kyber128_ineq_num[:indices], np.array(Kyber128_ineq_ps)[:indices], 'o', markersize=5, markerfacecolor='none', linestyle='dashed',
             linewidth=1, label='Probabilistic Statistic', color='green')
    plt.plot(Kyber128_ineq_num[:indices], np.array(Kyber128_ineq_com)[:indices], 's', markersize=5,
             markerfacecolor='none', linestyle='dashed',
             linewidth=1, label='Combinatorial Attack', color='red')

    # Labels and title
    plt.xlabel("Number of inequality hints")
    plt.ylabel("BKZ-$\\beta$")
    plt.legend(loc='upper right')

    plt.show()


def plot_Kyber128_perfect_hint_Com():
    Kyber128_perfect_num = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    Kyber128_perfect_emb = [50.58, 41.76, 32.17, 22.52, 14.84, 9.72, 6.28, 3.64, 2, 2, 2, 2, 2, 2]
    Kyber128_perfect_ps = [50.58, 50.56, 50.49, 50.32, 49.79, 49.63, 49.00, 48.61, 48.23, 47.81, 47.48, 42.14, 35.66,
                           28.90, 23.48, 19.29, 13.67, 4.74, 2.00, 2.00]
    Kyber128_perfect_com = [50.58, 41.67, 31.92, 22.04, 14.41, 9.27, 5.41, 3.20, 2]

    # 只取横坐标在200及以下的值
    Kyber128_perfect_num = np.array(Kyber128_perfect_num)
    indices = np.max(np.nonzero(Kyber128_perfect_num <= 40)[0])


    # 画图
    plt.figure(figsize=(8, 6))
    plt.plot(Kyber128_perfect_num[:indices], np.array(Kyber128_perfect_emb)[:indices], 'v', markersize=5, markerfacecolor='none', linestyle='dashed',
             linewidth=1, label='Lattice Reduction', color='blue')
    plt.plot(Kyber128_perfect_num[:indices], np.array(Kyber128_perfect_ps)[:indices], 'o', markersize=5, markerfacecolor='none', linestyle='dashed',
             linewidth=1, label='Probabilistic Statistic', color='green')
    plt.plot(Kyber128_perfect_num[:indices], np.array(Kyber128_perfect_com)[:indices], 's', markersize=5,
             markerfacecolor='none', linestyle='dashed',
             linewidth=1, label='Combinatorial Attack', color='red')

    # Labels and title
    plt.xlabel("Number of perfect hints")
    plt.ylabel("BKZ-$\\beta$")
    plt.legend(loc='upper right')

    plt.show()


def plot_Kyber128_modular_hint_LR_PS():
    Kyber128_modular_num = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
    Kyber128_modular_emb = [50.58, 43.67, 36.67, 29.29, 22.74, 18.01, 13.02, 9.23, 6.13, 3.57, 2, 2, 2, ]
    Kyber128_modular_ps = [50.58, 50.58, 50.58, 50.58, 50.58, 50.58, 50.58, 50.58, 50.58, 50.58, 50.58, 50.58, 50.58]

    # 只取横坐标在200及以下的值
    Kyber128_modular_num = np.array(Kyber128_modular_num)
    indices_LR = np.max(np.nonzero(Kyber128_modular_num <= 60)[0])
    indices_PS = np.max(np.nonzero(Kyber128_modular_num <= 60)[0])


    # 画图
    plt.figure(figsize=(8, 6))
    plt.plot(Kyber128_modular_num[:indices_LR], np.array(Kyber128_modular_emb)[:indices_LR], 'v', markersize=5, markerfacecolor='none', linestyle='dashed',
             linewidth=1, label='Lattice Reduction [DDGR20]', color='blue')
    plt.plot(Kyber128_modular_num[:indices_PS], np.array(Kyber128_modular_ps)[:indices_PS], 'o', markersize=5, markerfacecolor='none', linestyle='dashed',
             linewidth=1, label='Probabilistic Statistic', color='green')

    # Labels and title
    plt.xlabel("Kyber128 modular Num")
    plt.ylabel("Values")
    plt.title("Comparison of Kyber128 modular Hints")
    plt.legend()

    plt.show()


def plot_Kyber128_approx_hint_dis():
    Kyber128_approx_num = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200,
                           210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380,
                           390, 400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560,
                           570, 580, 590, 600, 610, 620, 630, 640, 650]

    Kyber128_approx_dis = [20.42, 20.36, 20.1, 19.76, 19.44, 19, 18.61, 18.1, 17.86, 17.42, 17.03, 16.52, 16.19, 15.79,
                           15.51, 14.97, 14.64, 14.13, 13.95, 13.44, 13.09, 12.89, 12.47, 12.04, 11.69, 11.48, 11.19,
                           10.82, 10.48, 10.42, 9.98, 9.83, 9.46, 9.31, 9.05, 8.76, 8.51, 8.18, 8.03, 7.63, 7.32, 7.15,
                           6.58, 6.33, 6.08, 5.37, 4.91, 4.41, 3.6, 2.97, 2.66, 1.84, 1.8, 1.08, 1.15, 0.79, 0.58, 0.37,
                           0.11, 0.1, 0.14, 0.06, 0.04, 0.03, 0.01, 0.0]

    # 只取横坐标在200及以下的值
    Kyber128_approx_num = np.array(Kyber128_approx_num)
    indices = np.max(np.nonzero(Kyber128_approx_num <= 650)[0])
    print("indices", indices)

    # 画图
    plt.figure(figsize=(8, 6))
    plt.plot(Kyber128_approx_num[:indices], np.array(Kyber128_approx_dis)[:indices],
             linewidth=1, label='Probabilistic Statistic', color='blue')

    # Labels and title
    plt.xlabel("Num of approximate hints", fontdict={'family': 'Times New Roman', 'size': 14})
    plt.ylabel(r"Norm($\tilde{s}-s$)", fontdict={'family': 'Times New Roman', 'size': 14})
    plt.title("The distance between guessed value with the secret key for LWE_128 ", fontdict={'family': 'Times New Roman', 'size': 14})
    plt.legend(prop={'size': 14})

    plt.show()


def plot_Kyber128_LR():
    Kyber128_ineq_num = [0, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680, 720, 760, 800]
    Kyber128_ineq_emb = [50.58, 47.70, 45.79, 44.42, 43.52, 42.97, 42.01, 40.99, 40.16, 39.43, 38.67, 38.10, 37.76, 37.42, 37.06, 36.73, 36.51, 36.29, 35.90, 36.55, 36.39]
    Kyber128_ineq_num = np.array(Kyber128_ineq_num)
    indices_ineq = np.max(np.nonzero(Kyber128_ineq_num <= 350)[0])

    Kyber128_approx_num = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200,
                           210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380,
                           390, 400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560,
                           570, 580, 590, 600, 610, 620, 630, 640, 650]
    Kyber128_approx_emb = [50.58, 46.37, 42.19, 37.96, 33.7, 29.26, 24.43, 22.05, 19.55, 16.5, 14.13, 12.48, 10.71,
                           9.46, 8.56, 7.68, 6.72, 5.64, 4.82, 4.44, 4.06, 3.61, 3.15, 2.5, 2, 2]
    Kyber128_approx_num = np.array(Kyber128_approx_num)
    indices_approx = np.max(np.nonzero(Kyber128_approx_num <= 250)[0])

    Kyber128_perfect_num = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    Kyber128_perfect_emb = [50.58, 41.76, 32.17, 22.52, 14.84, 9.72, 6.28, 3.64, 2, 2, 2, 2, 2, 2]
    Kyber128_perfect_num = np.array(Kyber128_perfect_num)
    indices_perfect = np.max(np.nonzero(Kyber128_perfect_num <= 50)[0])

    Kyber128_modular_num = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
    Kyber128_modular_emb = [50.58, 43.67, 36.67, 29.29, 22.74, 18.01, 13.02, 9.23, 6.13, 3.57, 2, 2, 2, ]
    Kyber128_modular_num = np.array(Kyber128_modular_num)
    indices_modular = np.max(np.nonzero(Kyber128_modular_num <= 60)[0])


    # 画图
    plt.figure(figsize=(8, 6))
    plt.plot(Kyber128_ineq_num[:indices_ineq], np.array(Kyber128_ineq_emb)[:indices_ineq], 'v', markersize=5, markerfacecolor='none', linestyle='dashed',
             linewidth=1, label='Inequality hints', color='blue')
    plt.plot(Kyber128_approx_num[:indices_approx], np.array(Kyber128_approx_emb)[:indices_approx], 'o', markersize=5, markerfacecolor='none',
             linestyle='dashed',
             linewidth=1, label='Approximate hints', color='green')
    plt.plot(Kyber128_modular_num[:indices_modular], np.array(Kyber128_modular_emb)[:indices_modular], '*',
             markersize=5, markerfacecolor='none',
             linestyle='dashed',
             linewidth=1, label='Modular hints', color='orange')
    plt.plot(Kyber128_perfect_num[:indices_perfect], np.array(Kyber128_perfect_emb)[:indices_perfect], 's', markersize=5, markerfacecolor='none',
             linestyle='dashed',
             linewidth=1, label='Perfect hints', color='red')


    # Labels and title
    plt.xlabel("Number of hints")
    plt.ylabel("BKZ-$\\beta$")
    plt.legend(loc='upper right')

    plt.show()


def plot_Kyber128_PS():
    Kyber128_ineq_num = [0, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680, 720, 760, 800]
    Kyber128_ineq_ps = [50.58, 50.58, 50.58, 49.29, 47.78, 45.78, 45.08, 42.72, 40.26, 37.08, 35.09, 29.91, 28.19, 22.72, 21.32, 10.66, 4.76, 3.23, 2, 2, 2]
    Kyber128_ineq_num = np.array(Kyber128_ineq_num)
    indices_ineq = np.max(np.nonzero(Kyber128_ineq_num <= 800)[0])

    Kyber128_approx_num = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200,
                           210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380,
                           390, 400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560,
                           570, 580, 590, 600, 610, 620, 630, 640, 650]
    Kyber128_approx_ps = [50.58, 50.45, 49.89, 49.14, 48.43, 47.43, 46.52, 45.31, 44.72, 43.62, 42.63, 41.29, 40.40, 39.29, 38.49, 36.91, 35.89, 34.27, 33.68, 31.95, 30.69, 29.95, 28.31, 26.13, 24.66, 24.19, 23.53, 22.64, 21.80, 21.65, 20.49, 20.08, 18.85, 18.31, 17.34, 16.18, 15.08, 14.31, 13.97, 13.00, 12.19, 11.71, 9.97, 9.56, 9.14, 7.77, 6.69, 5.15, 4.00, 2.76, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00]
    Kyber128_approx_num = np.array(Kyber128_approx_num)
    indices_approx = np.max(np.nonzero(Kyber128_approx_num <= 550)[0])

    Kyber128_perfect_num = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    Kyber128_perfect_ps = [50.58, 50.56, 50.49, 50.32, 49.79, 49.63, 49.00, 48.61, 48.23, 47.81, 47.48, 42.14, 35.66, 28.90, 23.48, 19.29, 13.67, 4.74, 2.00, 2.00]
    Kyber128_perfect_num = np.array(Kyber128_perfect_num)
    indices_perfect = np.max(np.nonzero(Kyber128_perfect_num <= 500)[0])

    Kyber128_modular_num = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600]
    Kyber128_modular_ps = [50.58, 50.5, 50.4, 50.3, 50.3, 50.3, 50.3, 50.3, 50.3, 50.3, 50.3, 50.3, 50.3]
    Kyber128_modular_num = np.array(Kyber128_modular_num)
    indices_modular = np.max(np.nonzero(Kyber128_modular_num <= 600)[0])


    # 画图
    plt.figure(figsize=(8, 6))
    plt.plot(Kyber128_ineq_num[:indices_ineq], np.array(Kyber128_ineq_ps)[:indices_ineq], 'v', markersize=5, markerfacecolor='none', linestyle='dashed',
             linewidth=1, label='Inequality hints', color='blue')
    plt.plot(Kyber128_approx_num[:indices_approx], np.array(Kyber128_approx_ps)[:indices_approx], 'o', markersize=5, markerfacecolor='none',
             linestyle='dashed',
             linewidth=1, label='Approximate hints', color='green')
    plt.plot(Kyber128_perfect_num[:indices_perfect], np.array(Kyber128_perfect_ps)[:indices_perfect], 's', markersize=5, markerfacecolor='none',
             linestyle='dashed',
             linewidth=1, label='Perfect hints', color='red')
    plt.plot(Kyber128_modular_num[:indices_modular], np.array(Kyber128_modular_ps)[:indices_modular], '*', markersize=5, markerfacecolor='none',
             linestyle='dashed',
             linewidth=1, label='Modular hints', color='orange')

    # Labels and title
    plt.xlabel("Kyber128 ineq Num")
    plt.ylabel("Values")
    plt.title("Comparison of Kyber128 ineq Hints")
    plt.legend()

    plt.show()

def plot_Kyber128_ineq_hint_dis():
    Kyber128_ineq_num = [0, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680, 720, 760,
                         800]
    Kyber128_ineq_dis = [18.52, 18.52, 18.52, 18.52, 17.89, 17.09, 16.82, 15.94, 15.07, 14.03, 13.35, 12, 11.58, 10.34,
                         9.8, 6.56, 4.24, 3.61, 0, 0, 0]

    # 只取横坐标在200及以下的值
    Kyber128_ineq_num = np.array(Kyber128_ineq_num)
    indices = np.max(np.nonzero(Kyber128_ineq_num <= 800)[0])
    print("indices", indices)

    # 画图
    plt.figure(figsize=(8, 6))
    plt.plot(Kyber128_ineq_num[:indices], np.array(Kyber128_ineq_dis)[:indices],
             linewidth=1, label='Probabilistic Statistic', color='blue')

    # Labels and title
    plt.xlabel("Num of inequality hints", fontdict={'family': 'Times New Roman', 'size': 14})
    plt.ylabel(r"Norm($\tilde{s}-s$)", fontdict={'family': 'Times New Roman', 'size': 14})
    plt.title("The distance between guessed value with the secret key for LWE_128 ",
              fontdict={'family': 'Times New Roman', 'size': 14})
    plt.legend(prop={'size': 14})

    plt.show()



def plot_Kyber128_perfect_hint_dis():
    Kyber128_perfect_num = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    Kyber128_perfect_dis = [20.02, 20.01, 19.98, 19.9, 19.66, 19.59, 19.31, 19.14, 18.97, 18.79, 18.65, 16.51, 14.28,
                            12.37, 10.95, 9.4, 7.75, 4.07, 1.16, 0]

    # 只取横坐标在200及以下的值
    Kyber128_perfect_num = np.array(Kyber128_perfect_num)
    indices = np.max(np.nonzero(Kyber128_perfect_num <= 500)[0])
    print("indices", indices)

    # 画图
    plt.figure(figsize=(8, 6))
    plt.plot(Kyber128_perfect_num[:indices], np.array(Kyber128_perfect_dis)[:indices],
             linewidth=1, label='Probabilistic Statistic', color='blue')

    # Labels and title
    plt.xlabel("Num of perfect hints", fontdict={'family': 'Times New Roman', 'size': 14})
    plt.ylabel(r"Norm($\tilde{s}-s$)", fontdict={'family': 'Times New Roman', 'size': 14})
    plt.title("The distance between guessed value with the secret key for LWE_128 ",
              fontdict={'family': 'Times New Roman', 'size': 14})
    plt.legend(prop={'size': 14})

    plt.show()


def plot_Kyber128_ps_dis():
    Kyber128_approx_num = [0,  30,  60,  90, 120,  150, 180, 210,  240, 270,  300, 330, 360, 390, 420, 450, 480,  510,  540, 570,  600,  630]
    Kyber128_approx_dis = [20.42,  19.76,  18.61,  17.42,  16.19, 14.97,  13.95,  12.89, 11.69, 10.82,  9.98,  9.31,  8.51, 7.63,  6.58,  5.37, 3.6,  1.84,  1.15,  0.37, 0.14,  0.03]

    Kyber128_ineq_num = [0, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680, 720]
    Kyber128_ineq_dis = [20.42, 20.42, 20.42, 20.42, 19.89, 18.09, 17.82, 15.94, 15.07, 14.03, 13.35, 12, 11.58, 10.34,
                         9.8, 6.56, 4.24, 3.61, 0, ]

    Kyber128_perfect_num = [0,  10,  20,  30, 40,  50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    Kyber128_perfect_dis = [20.42, 19.98, 19.66,  19.31,  18.97,  18.65, 16.51, 14.28,
                            12.37, 10.95, 9.4, 7.75, 4.07, 1.16, 0]

    # 画图
    plt.figure(figsize=(8, 6))
    plt.plot(Kyber128_ineq_num, np.array(Kyber128_ineq_dis), 'v', markersize=5,
             markerfacecolor='none', linestyle='dashed',
             linewidth=1, label='Inequality hints', color='blue')
    plt.plot(Kyber128_approx_num, np.array(Kyber128_approx_dis), 'o', markersize=5,
             markerfacecolor='none',
             linestyle='dashed',
             linewidth=1, label='Approximate hints', color='green')
    plt.plot(Kyber128_perfect_num, np.array(Kyber128_perfect_dis), 's',
             markersize=5, markerfacecolor='none',
             linestyle='dashed',
             linewidth=1, label='Perfect hints', color='red')
    # Labels and title
    plt.xlabel("Number of hints")
    plt.ylabel("Norm($\\tilde{s}-s$)")
    plt.legend(loc='upper right')


    plt.show()
    
def plot_Kyber80_lr_dis():
    Kyber80_approx_num = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
    Kyber80_approx_dis = [1367.16, 1504.44, 1854.34, 1747.37, 1919.91, 2132.70, 2336.85, 2084.36, 2736.61, 2918.64, 3033.99, 3711.85, 3315.71, 109.45, 0.00, 0.00, 0.00]

    Kyber80_ineq_num = []
    Kyber80_ineq_dis = []

    Kyber80_perfect_num = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    Kyber80_perfect_dis = [1453.05, 1464.70, 1535.88, 1594.48, 1741.73, 1735.91, 2263.58, 2236.55, 2365.23, 0.00, 0.00]

    # 画图
    plt.figure(figsize=(8, 6))
    plt.plot(Kyber80_perfect_num, Kyber80_perfect_dis, linewidth=1, label='Perfect hints', color='red')
    plt.plot(Kyber80_approx_num, Kyber80_approx_dis,linewidth=1, label='Approximate hints', color='blue')
    # plt.plot(Kyber80_ineq_num, Kyber80_ineq_dis, linewidth=1, label='Inequality hints', color='green')


    # Labels and title
    plt.xlabel("Num of LWE hints", fontdict={'family': 'Times New Roman', 'size': 14})
    plt.ylabel(r"Norm($\tilde{s}-s$)", fontdict={'family': 'Times New Roman', 'size': 14})
    plt.title("The distance between guessed value and the secret key for LWE_80 with BKZ_β=8", fontdict={'family': 'Times New Roman', 'size': 14})
    plt.legend(prop={'size': 14})

    plt.show()

def plot_Kyber128_lr_dis():
    Kyber80_approx_num = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
    Kyber80_approx_dis = [1367.16, 1504.44, 1854.34, 1747.37, 1919.91, 2132.70, 2336.85, 2084.36, 2736.61, 2918.64, 3033.99, 3711.85, 3315.71, 109.45, 0.00, 0.00, 0.00]

    Kyber80_ineq_num = []
    Kyber80_ineq_dis = []

    Kyber80_perfect_num = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    Kyber80_perfect_dis = [1453.05, 1464.70, 1535.88, 1594.48, 1741.73, 1735.91, 2263.58, 2236.55, 2365.23, 0.00, 0.00]

    # 画图
    plt.figure(figsize=(8, 6))
    plt.plot(Kyber80_perfect_num, Kyber80_perfect_dis, linewidth=1, label='Perfect hints', color='red')
    plt.plot(Kyber80_approx_num, Kyber80_approx_dis,linewidth=1, label='Approximate hints', color='blue')
    # plt.plot(Kyber80_ineq_num, Kyber80_ineq_dis, linewidth=1, label='Inequality hints', color='green')


    # Labels and title
    plt.xlabel("Num of LWE hints", fontdict={'family': 'Times New Roman', 'size': 14})
    plt.ylabel(r"Norm($\tilde{s}-s$)", fontdict={'family': 'Times New Roman', 'size': 14})
    plt.title("The distance between guessed value and the secret key for LWE_80 with BKZ_β=8", fontdict={'family': 'Times New Roman', 'size': 14})
    plt.legend(prop={'size': 14})

    plt.show()



def plot_Kyber256_approx_hint():
    Kyber256_approx_num = [0, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680, 720,
                           760, 800, 840, 880, 920, 960, 1000]
    Kyber256_approx_emb = [165.1, 151.78, 139.88, 129.01, 119.73, 110.94, 103.4, 96.37, 90.04, 84.16, 78.7, 73.68,
                           69.16, 65.43, 62.44, 59.99, 57.96, 56.16, 54.58, 53.14, 51.82, 50.52, 49.38, 48.29, 47.28,
                           46.34]
    Kyber256_approx_dis = [26.91, 26.39, 25.45, 24.6, 23.65, 22.08, 21.07, 20.21, 18.78, 18.23, 16.86, 16.03, 15.29,
                           14.36, 13.62, 13.1, 11.79, 11.05, 10.44, 9.17, 6.05, 4.61, 1.08, 0.97, 0, 0]
    Kyber256_approx_ps = [165.10, 163.86, 161.58, 159.48, 157.08, 153.01, 150.31, 147.96, 143.92, 142.32, 138.22,
                          135.64, 133.28, 130.22, 127.70, 125.88, 121.10, 118.26, 115.84, 110.50, 95.01, 86.03, 46.24,
                          43.35, 2.00, 2.00]
    Kyber256_approx_com = [165.10, 150.66, 137.00, 125.08, 114.22, 103.46, 94.62, 86.83, 78.84, 72.46, 65.09, 58.81,
                           53.10, 47.40, 42.52, 38.58, 32.97, 28.73, 24.51, 21.72, 13.95, 10.49, 3.26, 2.73, 2.00, 2.00]

    # 只取横坐标在800及以下的值
    Kyber256_approx_num = np.array(Kyber256_approx_num)
    mask = Kyber256_approx_num <= 760

    # 画图
    plt.figure(figsize=(8, 6))
    plt.plot(Kyber256_approx_num[mask], np.array(Kyber256_approx_emb)[mask], 'v', markersize=5, markerfacecolor='none', linestyle='dashed',
             linewidth=1, label='Lattice Reduction [DDGR20]', color='blue')
    plt.plot(Kyber256_approx_num[mask], np.array(Kyber256_approx_ps)[mask], 'o', markersize=5, markerfacecolor='none', linestyle='dashed',
             linewidth=1, label='Probabilistic Statistic', color='green')
    plt.plot(Kyber256_approx_num[mask], np.array(Kyber256_approx_com)[mask], 's', markersize=5, markerfacecolor='none', linestyle='dashed',
             linewidth=1, label='Combinatorial Attack', color='red')

    # Labels and title
    plt.xlabel("Kyber256 Approx Num")
    plt.ylabel("Values")
    plt.title("Comparison of Kyber256 Approximation Hints")
    plt.legend()

    plt.show()


def plot_Kyber256_perfect_hint():
    Kyber256_perfect_num = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190,
                            200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100]
    Kyber256_perfect_emb = [165.1, 151.71, 138.93, 126.74, 115.08, 103.92, 93.23, 82.95, 72.99, 63.2, 53.22, 42.28,
                            30.14, 20.46, 13.11, 8.69, 5.05, 3.23, 2, 2, 2]
    Kyber256_perfect_dis = [27.35, 27.33, 27.3, 27.16, 27.02, 26.73, 26.57, 26.32, 25.95, 25.87, 25.5, 25.15, 25.03,
                            24.75, 24.43, 24.34, 24.02, 23.58, 23.41, 23.01, 22.84, 20.1, 17.2, 15.37, 13.71, 12.23,
                            9.43, 3.69, 0.11, 0]
    Kyber256_perfect_ps = [165.10, 165.05, 164.98, 164.66, 164.33, 163.64, 163.26, 162.67, 161.78, 161.59, 160.69,
                           159.84, 159.54, 158.86, 158.06, 157.84, 157.04, 155.93, 155.50, 154.48, 154.04, 146.8, 138.4,
                           132.7, 127.2, 122.0, 111.0, 78.74, 8.67, 2.00]
    Kyber256_perfect_com = [165.10, 151.66, 138.83, 126.37, 114.51, 102.98, 92.15, 80.64, 70.34, 60.53, 50.00, 38.37,
                            24.87, 17.50, 11.15, 7.53, 4.41, 2.29, 2, 2, 2]

    # 只取横坐标在200及以下的值
    Kyber256_perfect_num = np.array(Kyber256_perfect_num)
    indices = np.max(np.nonzero(Kyber256_perfect_num <= 150)[0])
    print("indices", indices)

    # 画图
    plt.figure(figsize=(8, 6))
    plt.plot(Kyber256_perfect_num[:indices], np.array(Kyber256_perfect_emb)[:indices], 'v', markersize=5, markerfacecolor='none', linestyle='dashed',
             linewidth=1, label='Lattice Reduction [DDGR20]', color='blue')
    plt.plot(Kyber256_perfect_num[:indices], np.array(Kyber256_perfect_ps)[:indices], 'o', markersize=5, markerfacecolor='none', linestyle='dashed',
             linewidth=1, label='Probabilistic Statistic', color='green')
    plt.plot(Kyber256_perfect_num[:indices], np.array(Kyber256_perfect_com)[:indices], 's', markersize=5, markerfacecolor='none', linestyle='dashed',
             linewidth=1, label='Combinatorial Attack', color='red')

    # Labels and title
    plt.xlabel("Kyber256 perfect Num")
    plt.ylabel("Values")
    plt.title("Comparison of Kyber256 perfect Hints")
    plt.legend()

    plt.show()


def plot_Kyber512_dfa_hint():
    Kyber512_dfa_num = [0, 2, 4, 6, 8, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170]
    Kyber512_dfa_emb = [405.53, 404.88, 404.23, 403.59, 402.96, 402.32, 399.19, 396.11, 393.08, 390.12, 387.19, 384.27,
                        381.41, 378.58, 375.8, 373.05, 370.35, 367.66, 365.03, 362.42, 359.85, 357.3]
    Kyber512_dfa_dis = [40.09, 32.48, 28.04, 25.10, 22.78, 21.31, 17.28, 14.58, 12.54, 11.38, 10.01, 9.06, 8.14, 7.25,
                        6.28, 5.37, 4.22, 3.27, 2.42, 1.56, 0.77, 0.23]
    Kyber512_dfa_ps = [405.53, 380.68, 364.64, 353.19, 343.60, 337.22, 318.25, 303.99, 292.09, 284.78, 275.51, 268.60,
                       261.45, 254.01, 245.20, 236.07, 222.91, 210.05, 196.14, 177.97, 153.08, 119.67]
    Kyber512_dfa_com = [405.53, 380.09, 363.52, 351.57, 341.49, 334.67, 313.68, 297.65, 284.2, 275.36, 264.82, 256.64,
                        248.38, 239.98, 230.43, 220.73, 207.51, 194.72, 181.13, 163.84, 140.62, 109.71]

    # 只取横坐标在800及以下的值
    Kyber512_dfa_num = np.array(Kyber512_dfa_num)
    mask = Kyber512_dfa_num <= 125

    # 画图
    plt.figure(figsize=(8, 6))
    plt.plot(Kyber512_dfa_num[mask], np.array(Kyber512_dfa_emb)[mask], 'v', markersize=5, markerfacecolor='none', linestyle='dashed',
             linewidth=1, label='Lattice Reduction', color='blue')
    plt.plot(Kyber512_dfa_num[mask], np.array(Kyber512_dfa_ps)[mask], 'o', markersize=5, markerfacecolor='none', linestyle='dashed',
             linewidth=1, label='Probabilistic Statistic', color='green')
    plt.plot(Kyber512_dfa_num[mask], np.array(Kyber512_dfa_com)[mask], 's', markersize=5, markerfacecolor='none', linestyle='dashed',
             linewidth=1, label='Combinatorial Attack', color='red')


    # Labels and title
    plt.xlabel("Number of hints")
    plt.ylabel("BKZ-$\\beta$")
    plt.legend(loc='upper right')

    plt.show()

# PLOT the relation of num of ineqs and recovered coes
def coe_ine_512():
    ine_Bay = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000, 16000]
    coe_Bay = [141, 342.9, 433.2, 490.8, 536.8, 567.1, 604.9, 626.9, 660.1, 676.6, 701.5, 723.8, 739.2, 757.1, 774.4, 785.3, 796.9, 812.4, 827.5, 835.6, 842, 962, 998, 1012, 1019, 1022, 1023, 1023, 1023, 1023, 1023, 1024, 1024, 1024, 1024, 1024]

    ine_maj = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950,
               1000, 2000, 3000, 4000, 5000, 6000]
    coe_maj = [70, 214.1, 245.3, 273, 293.8, 313.3, 331.3, 338.4, 350.7, 362.1, 384, 390.1, 411.4, 420.4, 430.6, 443.9, 448.9, 456, 461.8, 466.9,
               471.4, 507, 511.9, 512, 512, 512]

    # 使用'cubic'插值创建平滑函数
    f1 = interp1d(ine_Bay, coe_Bay, kind='cubic')
    f2 = interp1d(ine_maj, coe_maj, kind='cubic')

    # 生成细分的x值以绘制平滑曲线
    xnew1 = np.linspace(1, 16000, 16000)
    xnew2 = np.linspace(1, 6000, 6000)

    # 设置坐标轴纵轴长度为 1000
    plt.ylim(0, 1100)
    plt.xlim(-500, 16000)
    # 绘制平滑曲线
    plt.plot(xnew1, f1(xnew1), '-', color='orange', label='Probability Method [25]')
    plt.plot(xnew2, f2(xnew2), '-', color='g', label='Our Method')

    # 添加y=1024的红色横线
    plt.axhline(y=1026, color='r', linestyle='--')
    plt.plot([-500, 6000], [514, 514], color='r', linestyle='--')
    plt.plot([3000, 3000], [0, 512], color='blue', linestyle='--')
    plt.plot([13000, 13000], [0, 1024], color='blue', linestyle='--')

    # plt.axvline(x=3000, ymin=0, ymax=512, color='b', linestyle='--', label='x = 3000')

    # 在图中添加标注
    plt.text(1, 1026, 'y = 1024', color='red', verticalalignment='bottom', fontsize=14)
    plt.text(1, 514, 'y = 512', color='red', verticalalignment='bottom', fontsize=14)

    # 添加坐标轴标签
    plt.xlabel('num of failures', fontdict={'family': 'Times New Roman', 'size': 14})
    plt.ylabel('recovered coefficients', fontdict={'family': 'Times New Roman', 'size': 14})

    plt.legend(prop={'size': 12})
    plt.show()


if __name__ == '__main__':
    #coe_ine_512()
    # Section 3 Performance of Lattice Reduction on LWE with Hints
    # plot_Kyber128_LR()

    # Section 4 Performance of Probabilities Statistic on LWE with Hints
    # plot_Kyber128_ps_dis()

    # Section 5 Performance of Combinatorial Attack on LWE with Hints
    plot_Kyber128_approx_hint_Com()
    plot_Kyber128_ineq_hint_Com()
    plot_Kyber128_perfect_hint_Com()

    # Section 6 Experimental Validation
    # plot_LWE80_approx_hint()
    #plot_Kyber512_dfa_hint()


    #plot_Kyber128_LR()
    # plot_Kyber128_PS()
    #plot_Kyber128_approx_hint_LR_PS()
    #plot_Kyber128_ineq_hint_LR_PS()
    #plot_Kyber128_perfect_hint_LR_PS()
    #plot_Kyber128_modular_hint_LR_PS()

    # plot_Kyber80_lr_dis()

    # the relation of num of ineqs and recovered coes

    # plot_Kyber128_approx_hint()
    # plot_Kyber128_ineq_hint()
    # plot_Kyber128_perfect_hint()

    # plot_Kyber128_approx_hint_dis()
    # plot_Kyber128_ineq_hint_dis()
    # plot_Kyber128_perfect_hint_dis()
    # plot_Kyber128_ps_dis()
    #
    # plot_Kyber256_approx_hint()
    # plot_Kyber256_perfect_hint()






