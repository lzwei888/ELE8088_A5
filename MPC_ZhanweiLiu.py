import time
import pandas as pd
import control as ctrl
import numpy as np
import matplotlib.pyplot as plt
from qpsolvers import solve_qp

A = np.array([[0.983500, 2.782, 0],
              [-0.0006821, 0.978, 0],
              [-0.0009730, 2.804, 1]])
B = np.array([[0.01293],
              [0.00100],
              [0.001425]])


# function of qpsolvers, user could set the ratio of Q, N, Plot or not
def qpsolver(Q_r, N, init, solver, plot):
    # set Q ratio
    Q = np.array([[Q_r[0], 0, 0], [0, Q_r[1], 0], [0, 0, Q_r[2]]])
    R = np.array([1])
    Pf, _, K = ctrl.dare(A, B, Q, R)  # P_f

    # N = 200

    # set P
    QR = np.vstack((np.hstack((Q, np.zeros((3, 1)))), np.hstack((np.array([0, 0, 0]), R))))
    QR_arr = []
    for i in range(N):
        QR_arr.append(np.hstack((np.zeros((4, i * 4)), QR, np.zeros((4, 4 * (N - i - 1))))))

    P, i = QR_arr[0], 0
    while i < len(QR_arr) - 1:
        i += 1
        P = np.vstack((P, QR_arr[i]))

    P = np.vstack((np.hstack((P, np.zeros((4 * N, 3)))), np.hstack((np.zeros((3, 4 * N)), Pf))))  # P

    q = np.zeros((4 * N + 3, 1))  # q

    # set A
    ABI = np.hstack((A, B, -np.eye(3)))
    ABI_arr = []
    for i in range(N):
        ABI_arr.append(np.hstack((np.zeros((3, i * 4)), ABI, np.zeros((3, 4 * (N - i - 1))))))
    Az, i = ABI_arr[0], 0
    while i < len(ABI_arr) - 1:
        i += 1
        Az = np.vstack((Az, ABI_arr[i]))

    Az = np.vstack((np.hstack((np.eye(3), np.zeros((3, 4 * N)))), Az))  # A

    # initial state
    # init = np.array([[0.2007], [-0.01174], [0.43834]])  # [[0.2007], [-0.0021], [-0.16833]]
    b = np.vstack((init, np.zeros((3 * N, 1))))  # b

    # set G
    C = np.array([[1, 0, 0, 0],
                  [-1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, -1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, -1, 0],
                  [-1, 0, 1, 0],
                  [1, 0, -1, 0],
                  [0, 0, 0, 1],
                  [0, 0, 0, -1]])
    d = np.array([[0.2007], [0.2007], [0.2443], [0.2443], [0.6108], [0.6108], [0.4014], [0.4014], [0.4712], [0.4188]])

    G_arr = []
    for i in range(N):
        G_arr.append(np.hstack((np.zeros((10, i * 4)), C, np.zeros((10, 4 * (N - i - 1))))))
    G, i = G_arr[0], 0
    while i < len(G_arr) - 1:
        i += 1
        G = np.vstack((G, G_arr[i]))

    G = np.vstack((np.hstack((G, np.zeros((10 * N, 3)))), np.hstack((np.zeros((8, 4 * N)), C[0:8, :][:, 0:3]))))  # G

    h = d
    for i in range(N - 1):
        h = np.vstack((h, d))
    h = np.vstack((h, d[0:8, :]))  # h

    # set lb and ub
    zmin = np.array([[-0.2007], [-0.2443], [-0.6109], [-0.4189]])
    zmax = np.array([[0.2007], [0.2443], [0.6109], [0.4712]])
    z_min, z_max = zmin, zmax

    for i in range(N - 1):
        z_min = np.vstack((z_min, zmin))
        z_max = np.vstack((z_max, zmax))

    z_min = np.vstack((z_min, np.array([[-0.2007], [-0.2443], [-0.6109]])))
    z_max = np.vstack((z_max, np.array([[0.2007], [0.2443], [0.6109]])))

    ## result
    # print("size: ")
    # print("P: ", P.shape)
    # print("q: ", q.shape)
    # print("G: ", G.shape)
    # print("h: ", h.shape)
    # print("A: ", Az.shape)
    # print("b: ", b.shape)
    # print("z_min: ", z_min.shape)
    # solve and record time
    start = time.time()
    x = solve_qp(P, q, G, h, Az, b, lb=z_min, ub=z_max, solver=solver)  # lb=z_min, ub=z_max,
    end = time.time()
    t = end - start
    print("\ninitial state: \n", init.T[0], ".T")
    print(f"\nQP solution: \n", x)
    print("\ntime: ", t)

    alpha, q_rate, theta, u = [], [], [], []

    x1 = x[:N * 4].reshape(-1, 4)

    x2 = x[N * 4:]

    column_arr = [x1[:, i] for i in range(x1.shape[1])]

    alpha = np.append(column_arr[0], x2[0]).T
    q_rate = np.append(column_arr[1], x2[1]).T
    theta = np.append(column_arr[2], x2[2]).T
    u = column_arr[3].T

    # ux = np.arange(0, N * 0.05, 0.05)
    # plt.plot(ux, u, 'r')
    # plt.show()

    # plot
    x_axis_data = np.arange(0, N * 0.05 + 0.05, 0.05)  # [i for i in range(0, N + 1)]
    title = f'N = {N} , Q: {Q[0, 0]}, {Q[1, 1]}, {Q[2, 2]}'

    plt.subplot(3, 1, 1)
    plt.plot(x_axis_data, np.degrees(alpha), 'b-', alpha=0.5, linewidth=1, label=r'$\alpha$')
    plt.ylabel("degree")
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(x_axis_data, np.degrees(q_rate), 'r-', alpha=0.5, linewidth=1, label="q")
    plt.ylabel("degree")
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(x_axis_data, np.degrees(theta), 'g-', alpha=0.5, linewidth=1, label=r'$\theta$')
    # plt.plot(x_axis_data.pop(), u, 'y-', alpha=0.5, linewidth=1, label=r'$\delta')
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("degree")
    plt.suptitle(title)
    if plot:
        plt.show()
    return t


# qpsolvers
Q_r = [5000, 1100, 1500]  # select ratio of Q
N = 500  # select N
init = np.array([[0.2007], [-0.01174], [0.43834]])  # set initial state
solver = "piqp"  # select solvers
plot = True  # show plot?
qpsolver(Q_r, N, init, solver, plot)  # call qpsolver function

# # plot time vs N
# N, T = [4, 10, 30, 50, 100, 200, 300, 500], []
# for i in N:
#     t = qpsolver(Q_r, i, init, solver, plot)  # call qpsolver function
#     T.append(t)
# print(T)

# # average time and maximum time
# T = []
# for i in range(10):
#     t = qpsolver(Q_r, N, init, solver, plot)
#     T.append(t)
#
# print("average: ", np.mean(T))
# print("maximum: ", max(T))
