import pandas as pd
import control as ctrl
import numpy as np
from qpsolvers import solve_qp

'''
python is able to read excel in the format of np.array, and store arrays into excel.
Matrix P, A, G are read from P.xlsx, A.xlsx, G.xlsx
N is set to 4 for testing, therefore n = 19

## write excel 
# data_df = pd.DataFrame(G)
# writer = pd.ExcelWriter('G.xlsx')
# data_df.to_excel(writer, 'page_1', float_format='%.5f')
# writer.save()

## read excel to np.array
# df = pd.read_excel('G.xlsx', 'page_1', header=1)
# G = np.array(df)
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

# set N
N = 4

# set P
df = pd.read_excel('P.xlsx', 'page_1', header=1)
P = np.array(df)

q = np.zeros((19, 1))  # q

# set A
df = pd.read_excel('A.xlsx', 'page_1', header=1)
Az = np.array(df)

# initial state
init = np.array([[0.2007], [-0.0021], [-0.16833]])
b = np.vstack((init, np.zeros((12, 1))))  # b

# set G
df = pd.read_excel('G.xlsx', 'page_1', header=1)
G = np.array(df)

d = np.array([[0.2007], [0.2007], [0.2443], [0.2443], [0.6108], [0.6108], [0.4014], [0.4014], [0.4712], [0.4188]])
h = d
for i in range(N - 1):
    h = np.vstack((h, d))
h = np.vstack((h, d[0:8, :]))  # h

# result
x = solve_qp(P, q, G, h, Az, b, solver="proxqp")  # lb=z_min, ub=z_max,
print(f"QP solution: {x = }")

## write excel
# data_df = pd.DataFrame(G)
# writer = pd.ExcelWriter('G.xlsx')
# data_df.to_excel(writer, 'page_1', float_format='%.5f')
# writer.save()
