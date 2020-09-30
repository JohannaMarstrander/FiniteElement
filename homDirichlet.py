import numpy as np
from poisson2D import createAandF
from plot import plot


def homogeneousDirichlet(N, Nq, f):
    A, F, edge, p, tri = createAandF(f, N, Nq)
    nodes = np.unique(edge)
    F[nodes] = 0
    epsilon = 1e-16
    A[nodes, nodes] = 1 / epsilon
    u = np.linalg.solve(A, F)
    return u, p


def f(x, y):
    return -8 * np.pi * np.cos(2 * np.pi * (x ** 2 + y ** 2)) + 16 * np.pi ** 2 * (x ** 2 + y ** 2) * np.sin(
        2 * np.pi * (x ** 2 + y ** 2))


def u(x, y):
    return np.sin(2 * np.pi * (x ** 2 + y ** 2))


u_num, p = homogeneousDirichlet(300, 4, f)
u_ex = u(p[:, 0], p[:, 1])

plot(p[:, 0], p[:, 1], u_num)
plot(p[:, 0], p[:, 1], u_ex)
plot(p[:, 0], p[:, 1], u_num - u_ex)
