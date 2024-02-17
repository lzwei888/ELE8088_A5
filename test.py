import time
import pandas as pd
import control as ctrl
import polytope as pc
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as spla
from qpsolvers import solve_qp


N = 5
# set lb and ub
zmin = np.array([[-0.2007], [-0.2443], [-0.6109], [-0.4189]])
zmax = np.array([[0.2007], [0.2443], [0.6109], [0.4712]])
z_min, z_max = zmin, zmax

for i in range(N - 1):
    z_min = np.vstack((z_min, zmin))
    z_max = np.vstack((z_max, zmax))

z_min = np.vstack((z_min, np.array([[-0.2007], [-0.2443], [-0.6109]])))
z_max = np.vstack((z_max, np.array([[0.2007], [0.2443], [0.6109]])))
