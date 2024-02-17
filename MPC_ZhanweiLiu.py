import time
import pandas as pd
import control as ctrl
import polytope as pc
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as spla
from qpsolvers import solve_qp

'''
The following code and variables are used to create matrix G h A b P q, it s not the optimal code 
Q is I(4) and R is 1 in the code, but I think we need to change it according to the solution 
'''
A = np.array([[0.983500, 2.782, 0],
              [-0.0006821, 0.978, 0],
              [-0.0009730, 2.804, 1]])
B = np.array([[0.01293],
              [0.00100],
              [0.001425]])

Q = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
R = np.array([1])
Pf, _, K = ctrl.dare(A, B, Q, R)  # P_f

N = 100

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
init = np.array([[0.2007], [-0.0021], [-0.16833]])
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


# result
print("size: ")
print("P: ", P.shape)
print("q: ", q.shape)
print("G: ", G.shape)
print("h: ", h.shape)
print("A: ", Az.shape)
print("b: ", b.shape)
print("z_min: ", z_min.shape)
# solve and record time
start = time.time()
x = solve_qp(P, q, G, h, Az, b, lb=z_min, ub=z_max, solver="proxqp")  # lb=z_min, ub=z_max,
end = time.time()

print("\ninitial state: \n", init.T, ".T")
print(f"\nQP solution: \n", x)
print("\ntime: ", end - start)

alpha, q_rate, theta, u = [], [], [], []

x1 = x[:N * 4].reshape(-1, 4)

x2 = x[N * 4:]

column_arr = [x1[:, i] for i in range(x1.shape[1])]

alpha = np.append(column_arr[0], x2[0]).T
q_rate = np.append(column_arr[1], x2[1]).T
theta = np.append(column_arr[2], x2[2]).T
u = column_arr[3].T

x_axis_data = [i for i in range(0, N + 1)]  # x
plt.title("N = 100")
plt.plot(x_axis_data, alpha, 'b-', alpha=0.5, linewidth=1, label=r'$\alpha$')
plt.plot(x_axis_data, q_rate, 'r-', alpha=0.5, linewidth=1, label="q")
plt.plot(x_axis_data, theta, 'g-', alpha=0.5, linewidth=1, label=r'$\theta$')
# plt.plot(x_axis_data.pop(), u, 'y-', alpha=0.5, linewidth=1, label=r'$\delta')
plt.legend()
plt.show()


#
# data_df = pd.DataFrame(G)
# writer = pd.ExcelWriter('G.xlsx')
# data_df.to_excel(writer, 'page_1', float_format='%.5f')
# writer.save()

# df = pd.read_excel('P.xlsx', 'page_1', header=1)
# P = np.array(df)
# print(P)
