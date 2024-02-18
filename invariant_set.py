import control as ctrl
import polytope as pc
import numpy as np
import scipy.linalg as spla

# gamma is added to system dynamics
A = np.array([[0.983500, 2.782, 0],
              [-0.0006821, 0.978, 0],
              [-0.0009730, 2.804, 1]])
B = np.array([[0.01293],
              [0.00100],
              [0.001425]])

Q = np.eye(3)
R = np.array([1])

# sampling time of 0.05 seconds
dt = 0.05

# get matrix P_f
P, _, K = ctrl.dare(A, B, Q, R)
K = -K
Acl = A + B @ K

# set lb and ub, change deg to rad, gamma is added
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

X_1_extreme_points = pc.extreme(X_1)
extreme_points = pc.extreme(mis)
# ax = plt.figure().add_subplot(projection='3d')
# ax.scatter(extreme_points[:, 0], extreme_points[:, 1], zs=extreme_points[:, 2])
# ax.set_xlim(-0.5, 0.5)
# ax.set_ylim(-0.5, 0.5)
# ax.set_zlim(-0.3, 0.3)
# ax.set_xlabel('$x_1$')
# ax.set_ylabel('$x_2$')
# ax.set_zlabel('$x_3$')
# plt.show()


# np.set_printoptions(threshold=np.inf)
# print(extreme_points)