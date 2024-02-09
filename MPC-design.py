import time
import pandas as pd
import control as ctrl
import polytope as pc
import numpy as np
import control.optimal as obc
import matplotlib.pyplot as plt
import scipy.linalg as spla
from qpsolvers import solve_qp

# State space model parameters for aircraft pitch control
A = np.array([[0.983500, 2.782, 0],
              [-0.0006821, 0.978, 0],
              [-0.0009730, 2.804, 1]])
B = np.array([[0.01293],
              [0.00100],
              [0.001425]])
# Assuming all states are measurable
# Q = np.eye(3)
Q = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
# Assuming there is no direct path through
R = np.array([1])

# Convert the model to a discrete-time system with a sampling time of 0.05 seconds
dt = 0.05

P, _, K = ctrl.dare(A, B, Q, R)
K = -K
Acl = A + B @ K

x_min = np.array([[-0.2007], [-0.2443], [-0.6109]])
x_max = -x_min
u_min = np.array([-0.4189])
u_max = np.array([0.4712])

# Define H_x, b_x, H_u and b_u
H_x = np.vstack((np.eye(3), -np.eye(3)))
b_x = np.vstack((x_max, -x_min))
X = pc.Polytope(H_x, b_x)
H_u = np.array([[1], [-1]])
b_u = np.vstack((u_max, -u_min))

# Define X_kappa_f
H_kappa_f = np.vstack((H_x, H_u @ K))
b_kappa_f = np.vstack((b_x, b_u))
X_kappa_f = pc.Polytope(H_kappa_f, b_kappa_f)


def next_polytope(poly_j, poly_kappa_f, a_closed_loop):
    (Hj, bj) = (poly_j.A, poly_j.b)
    (Hkf, bkf) = (poly_kappa_f.A, poly_kappa_f.b)
    Hnext = np.vstack((Hkf, Hj @ a_closed_loop))
    bnext = np.concatenate((bkf, bj))
    return pc.Polytope(Hnext, bnext)


def determine_maximal_invariant_set(poly_kappa_f, a_closed_loop, max_iters=100):
    inv_prev = poly_kappa_f
    keep_running = True
    i = 0
    while keep_running:
        i = i + 1
        inv_next = next_polytope(inv_prev, poly_kappa_f, a_closed_loop)
        keep_running = not inv_prev <= inv_next
        inv_prev = inv_next
        if i > max_iters:
            raise Exception("failed to compute MIS ")
    return inv_next


mis = determine_maximal_invariant_set(X_kappa_f, Acl)

(H_infty, b_infty) = (mis.A, mis.b)

H1_last_row_block = np.hstack((H_infty @ A, H_infty @ B))
H_1_before = np.vstack((spla.block_diag(H_x, H_u), H1_last_row_block))
b_1_before = np.concatenate((b_x, b_u, np.reshape(b_infty, (-1, 1))))

S_1_before = pc.Polytope(H_1_before, b_1_before)
X_1 = S_1_before.project([1, 2])

# n = 20

g1 = np.append(1, np.zeros((1, 4)))
g2 = np.append(-1, np.zeros((1, 4)))
g3 = np.append(np.array([0, 1]), np.zeros((1, 3)))
g4 = np.append(np.array([0, -1]), np.zeros((1, 3)))
g5 = np.append(np.array([0, 0, 1]), np.zeros((1, 2)))
g6 = np.append(np.array([0, 0, -1]), np.zeros((1, 2)))
g7 = np.append(np.array([-1, 0, 1]), np.zeros((1, 2)))
g8 = np.append(np.array([1, 0, -1]), np.zeros((1, 2)))
g9 = np.array([0, 0, 0, 0, 1])
g10 = np.array([0, 0, 0, 0, -1])

gg = np.vstack((g1, g2, g3, g4, g5, g6, g7, g8, g9, g10))

gg1 = np.hstack((gg, np.zeros((10, 15))))
gg2 = np.hstack((np.zeros((10, 5)), gg, np.zeros((10, 10))))
gg3 = np.hstack((np.zeros((10, 10)), gg, np.zeros((10, 5))))
gg4 = np.hstack((np.zeros((10, 15)), gg))

G1 = np.vstack((gg1, gg2, gg3, gg4, np.zeros((8, 20))))
G2 = np.vstack((np.zeros((40, 4)),
                np.array([[1, 0, 0, 0], [-1, 0, 0, 0], [0, 1, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, -1, 0],
                          [-1, 0, 1, 0], [1, 0, -1, 0]])))
G = np.hstack((G1, G2))

h = np.array([[0.2007], [0.2007], [0.2443], [0.2443], [0.6109], [0.6109], [0.4014], [0.4014], [0.4712], [0.4189]])
h = np.vstack((h, h, h, h, np.array([[0.2007], [0.2007], [0.2443], [0.2443], [0.6109], [0.6109], [0.4014], [0.4014]])))

Azz = np.vstack(((np.hstack((A, np.zeros((3, 1)), B, np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0]]))),
                  (np.array([-1, 0, 1, -1, 0, 0, 0, 0, -1])))))
Az1 = np.hstack((Azz, np.zeros((4, 15))))
Az2 = np.hstack((np.zeros((4, 5)), Azz, np.zeros((4, 10))))
Az3 = np.hstack((np.zeros((4, 10)), Azz, np.zeros((4, 5))))
Az4 = np.hstack((np.zeros((4, 15)), Azz))

Az = np.vstack((Az1, Az2, Az3, Az4))
# initial state
bz = np.vstack((np.zeros((12, 1)), np.array([[-0.2], [0.1], [0.1], [-0.1]])))

zmin = np.array([[-0.2007], [-0.2443], [-0.6109], [-0.4014], [-0.4189]])
zmax = np.array([[0.2007], [0.2443], [0.6109], [0.4014], [0.4712]])
z_min = np.vstack((zmin, zmin, zmin, zmin, np.array([[-0.2007], [-0.2443], [-0.6109], [-0.4014]])))
z_max = np.vstack((zmax, zmax, zmax, zmax, np.array([[0.2007], [0.2443], [0.6109], [0.4712]])))

A = np.array([[0.983500, 2.782, 0, 0],
              [-0.0006821, 0.978, 0, 0],
              [-0.0009730, 2.804, 1, 0],
              [-1, 0, 1, -1]])
B = np.array([[0.01293],
              [0.00100],
              [0.001425],
              [0]])

Q = np.eye(4)
R = np.array([1])
Pf, _, K = ctrl.dare(A, B, Q, R)

P1 = np.hstack((np.eye(5), np.zeros((5, 19))))
P2 = np.hstack((np.zeros((5, 5)), np.eye(5), np.zeros((5, 14))))
P3 = np.hstack((np.zeros((5, 10)), np.eye(5), np.zeros((5, 9))))
P4 = np.hstack((np.zeros((5, 15)), np.eye(5), np.zeros((5, 4))))
P5 = np.hstack((np.zeros((4, 20)), Pf))
P = np.vstack((P1, P2, P3, P4, P5))

q = np.zeros((24, 1))

start=time.time()
x = solve_qp(P, q, G, h, Az, bz, lb=z_min, ub=z_max, solver="proxqp")
end=time.time()
print(f"QP solution: {x = }")
print("time: ", start-end)

# data_df = pd.DataFrame(Az)
# writer = pd.ExcelWriter('A.xlsx')  # 关键2，创建名称为hhh的excel表格
# data_df.to_excel(writer, 'page_1', float_format='%.5f')  # 关键3，float_format 控制精度，将data_df写到hhh表格的第一页中。若多个文件，可以在page_2中写入
# writer.save()


'''
# lb=z_min, ub=z_max,
Mz = np.array([[1.0, 2.0, 0.0], [0.0, 2.0, 3.0], [0.0, 0.0, 1.0]])
Pzz = Mz.T @ Mz
QR = np.vstack((np.hstack((Pzz, np.zeros((3, 1)))), np.array([0, 0, 0, 1])))

# QR = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 2, 0], [0, 0, 0, 1]])

I = np.eye(3)
daleth = np.vstack((np.hstack((A, B, -I, np.zeros((3, 8)))), np.hstack((np.zeros((3, 4)), A, B, -I, np.zeros((3, 4)))),
                    np.hstack((np.zeros((3, 8)), A, B, -I)), np.hstack((I, np.zeros((3, 12))))))
eta = np.vstack((np.zeros((11, 1)), 1))
G = np.hstack((np.vstack((H_1_before, np.zeros((4, 4)))), np.zeros((140, 11))))
h = np.vstack((b_1_before, np.zeros((4, 1))))
# Pz = np.vstack((np.hstack((np.eye(12), np.zeros((12, 3)))), np.hstack((np.zeros((3, 12)), P))))
Pz = np.vstack((np.hstack((QR, np.zeros((4, 11)))), np.hstack((np.zeros((4, 4)), QR, np.zeros((4, 7)))),
                np.hstack((np.zeros((4, 8)), QR, np.zeros((4, 3)))), np.hstack((np.zeros((3, 12)), P))))
# qz = np.ones((15, 1))
qz = np.vstack((np.zeros((14, 1)), 1))
zmin1 = np.vstack((x_min, u_min))
zmin = np.vstack((zmin1, zmin1, zmin1, x_min))
zmax1 = np.vstack((x_max, u_max))
zmax = np.vstack((zmax1, zmax1, zmax1, x_max))


def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)


if is_pos_def(Pz):
    print("yes")
    x = solve_qp(Pz, qz, G, h, daleth, eta, lb=zmin, ub=zmax, solver="proxqp")
    print(f"QP solution: {x = }")
else:
    print("Pz false")

# Pz = np.vstack((np.hstack((np.zeros((3, 137)), P)), np.hstack((np.eye(137), np.zeros((137, 3))))))
# qz = np.ones((140,1))'''
