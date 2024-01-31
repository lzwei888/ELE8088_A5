import time
import control as ctrl
import polytope as pc
import numpy as np
import control.optimal as obc
import matplotlib.pyplot as plt
import scipy.linalg as spla

# State space model parameters for aircraft pitch control
A = np.array([[0.983500, 2.782, 0],
              [-0.0006821, 0.978, 0],
              [-0.0009730, 0, 2.804]])
B = np.array([[0.01293],
              [0.00100],
              [0.001425]])
# Assuming all states are measurable
Q = np.eye(3)
# Assuming there is no direct path through
R = np.array([1])

# Convert the model to a discrete-time system with a sampling time of 0.05 seconds
dt = 0.05

P, _, K = ctrl.dare(A, B, Q, R)
K = -K
Acl = A + B @ K


x_max = np.array([[11.5],[14],[35]])
x_min = -x_max
u_max = np.array([27])
u_min = np.array([-24])

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



X_kappa_f.project([1, 2]).plot()

# X_kappa_f.plot(ax=X.plot(linestyle="-", color='yellow'), linestyle="-", color='pink')
# plt.legend((r"$X$", r"$X_{\kappa_{f}}$"), loc='upper right')
plt.xlim([-20, 20])
plt.ylim([-20, 20])



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
# print(mis)

mis.project([1, 2]).plot()

# X_kappa_f.plot(ax=X.plot(linestyle="-", color='yellow'), linestyle="-", color='pink')
# plt.legend((r"$X$", r"$X_{\kappa_{f}}$"), loc='upper right')
plt.xlim([-20, 20])
plt.ylim([-20, 20])
plt.show()

# # Here `mis` is the maximal invariant set
# extreme_points = pc.extreme(mis)
# ax = plt.figure().add_subplot(projection='3d')
# ax.scatter(extreme_points[:, 0], extreme_points[:, 1], zs=extreme_points[:, 2])
# ax.set_xlim(-10, 10)
# ax.set_ylim(-10, 10)
# ax.set_zlim(-10, 10)
# ax.set_xlabel('$x_1$')
# ax.set_ylabel('$x_2$')
# ax.set_zlabel('$x_3$')
# ax.view_init(elev=30., azim=30)
# plt.show()


(H_infty, b_infty) = (mis.A, mis.b)
# print("Hinf = \n", H_infty)
# print("binf = \n", b_infty)

H1_last_row_block = np.hstack((H_infty@A, H_infty@B))
H_1_before = np.vstack((spla.block_diag(H_x, H_u),H1_last_row_block))
b_1_before = np.concatenate((b_x, b_u, np.reshape(b_infty, (-1, 1))))

# print("Before the projection: {z: H_{z, 1}} z <= b_{z, 1}")
# print("where, H_{z, 1} = ", H_1_before)
# print("and b_{z, 1} = ", b_1_before)
S_1_before = pc.Polytope(H_1_before, b_1_before)
X_1 = S_1_before.project([1, 2])
# print("After: H_{1} = ", X_1.A)
# print("After: b_{1} = ", X_1.b)
# print(X_1 >= mis)

# # PLOT X_1 and O_infty
x1_plot_axes = X_1.project([1, 2]).plot(color="cyan", linestyle="-")
mis.project([1, 2]).plot(color="green", linestyle="--", ax=x1_plot_axes) \
    .legend((r"$X_{1}(O_{\infty})$", r"$O_{\infty}$"))
plt.xlim([-20, 20])
plt.ylim([-20, 20])
plt.show()

# The extreme points of X_1 are:
X_1_extreme_points = pc.extreme(X_1)


P_sqrt = spla.sqrtm(P)
n_constr_x_kappa_f = X_kappa_f.A.shape[0]
alpha = np.inf
for i in range(n_constr_x_kappa_f):
    z = np.linalg.solve(P_sqrt, X_kappa_f.A[i,:])
    norm_z_sq = np.linalg.norm(z, 2)**2
    bi = X_kappa_f.b[i]
    alpha = min(alpha, bi**2 / norm_z_sq)

th = np.linspace(0, 2 * np.pi, 200)
sth = np.sin(th)
cth = np.cos(th)
circle_points = np.sqrt(alpha) * np.vstack((sth, cth))
ellipsoid_points = np.linalg.solve(P_sqrt, circle_points)

# X_kappa_f.plot(ax=X.plot(linestyle="-", color='yellow'), linestyle="-", color='pink')
# plt.fill(ellipsoid_points[0, :], ellipsoid_points[1, :], alpha=0.7, facecolor='green', edgecolor='black', zorder=1)
# plt.legend((r"$X$", r"$X_{\kappa_{f}}$", r"$E_{\alpha}$"), loc='upper right')
#
# plt.xlim([-2.2, 2.2])
# plt.ylim([-2.2, 2.2])
# plt.xlabel("$x_{1}$")
# plt.ylabel("$x_{2}$")
# plt.show()




