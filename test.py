import time
import pandas as pd
import control as ctrl
import polytope as pc
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as spla
from qpsolvers import solve_qp


N = [4, 10, 30, 50, 100, 200, 300, 500]
T = [0.011002540588378906, 0.03900933265686035, 0.38408851623535156, 0.852196455001831,
     3.491804361343384, 18.314234733581543, 51.516892433166504, 204.53474116325378]
plt.title("Time for different N")
plt.plot(N, T, 'b-', alpha=0.5, linewidth=1)
plt.xlabel("N")
plt.ylabel("Time (s)")
plt.show()
