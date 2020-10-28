"""
Function for creating A and F with homogeneous Dirichlet boundary conditions.
Uses base function from poisson2D.
"""

import numpy as np
from poisson2D import createAandF


def homogeneousDirichlet(N, Nq, f):
    """
    Solving the poisson problem in 2D with homogeneous Dirichlet BCs.
    N is the number of nodes in triangulation, Nq is the number of 
    integration points in the gaussian quadrature. f is the rhs of the eq.
    Returns the solution u and a list of coordinates of nodes p.
    """
    A, F, edge, p, tri = createAandF(f, N, Nq)
    nodes = np.unique(edge)
    F[nodes] = 0
    epsilon = 1e-16
    A[nodes, nodes] = 1 / epsilon
    u = np.linalg.solve(A, F)
    return u, p



