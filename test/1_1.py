import time
import numpy as np
from scipy.sparse import csc_matrix
from qpsolvers import solve_qp
import warnings
warnings.filterwarnings("ignore")


# function qp_solve could call qpsolvers.solve_qp to solve convex quadratic optimi_Ation problems
def qp_solve(_P, _q, _G, _h, _A, _b, _xmin, _xmax, method):
    time_array = []
    for i in range(10):
        start = time.time()
        x = solve_qp(_P, _q, _G, _h, _A, _b, lb=_xmin, ub=_xmax, solver=method)
        end = time.time()
        time_array.append(end - start)
    print(method, "solver:\narray =", x.tolist())
    print("average computation times", np.array(time_array).mean())
    print("maximum computation times", np.array(time_array).max(), end='\n' * 2)


# boundary
xmin = np.array([-2.0, -2.0, -2.0])
xmax = np.array([2.0, 2.0, 2.0])

# numpy array
print("numpy.array: \n-------------")
M = np.array([[1.0, 2.0, 0.0], [0.0, 2.0, 3.0], [0.0, 0.0, 1.0]])
P = M.T @ M  # S^{3}_{++}
q = np.array([1.0, 2.0, 3.0]) @ M  # R^3
G = np.array([[-1.0, -2.0, -3.0], [1.0, 0.0, 1.0], [1.0, 2.0, 1.0]])  # R^{3x3}
h = np.array([3.0, 2.0, 1.0])  # R^3
A = np.array([1.0, 1.0, 1.0])
b = np.array([1.0])

qp_solve(P, q, G, h, A, b, xmin, xmax, "cvxopt")
qp_solve(P, q, G, h, A, b, xmin, xmax, "piqp")
qp_solve(P, q, G, h, A, b, xmin, xmax, "proxqp")

# scipy.sparse.csc_matrix
print("scipy.sparse.csc_matrix: \n-------------------------")
M = csc_matrix([[1.0, 2.0, 0.0], [0.0, 2.0, 3.0], [0.0, 0.0, 1.0]])
P = csc_matrix([[1, 2, 0], [2, 8, 6], [0, 6, 10]])  # S^{3}_{++}, P = M^T M
q = np.array([1.0, 2.0, 3.0]) @ M  # R^3
G = csc_matrix([[-1.0, -2.0, -3.0], [1.0, 0.0, 1.0], [1.0, 2.0, 1.0]])  # R^{3x3}
h = np.array([3.0, 2.0, 1.0])  # R^3
A = csc_matrix([1.0, 1.0, 1.0])
b = np.array([1.0])

qp_solve(P, q, G, h, A, b, xmin, xmax, "clarabel")
qp_solve(P, q, G, h, A, b, xmin, xmax, "piqp")
qp_solve(P, q, G, h, A, b, xmin, xmax, "proxqp")
