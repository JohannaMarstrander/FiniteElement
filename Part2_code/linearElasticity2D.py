from Part2_code.getplate import getPlate
import numpy as np
from Part1.Quadrature import quadrature2D
from scipy.sparse.linalg import spsolve
from scipy import sparse

def lin(x, y, c, g,*args):
    "utility function to create the F vecotr"
    return (c[0] + x * c[1] + y * c[2]) * g(x, y,*args)

def createAandF(f, N, Nq,nu,E):
    """Returns A, F,a list of corners of edge lines, a list of points, list of elements.
        f is the rhs of the eq, N^2 is number of nodes, Nq number of integration points in
        gaussian quadrature"""
    p, tri, edge = getPlate(N)
    A = np.zeros((2*N**2, 2*N**2))
    F = np.zeros(2*N**2)

    for el in tri:  # for each element
        # Find coefficients of H_alpha^k
        B = np.hstack((np.array([[1], [1], [1]]), p[el]))
        p1, p2, p3 = p[el[0]], p[el[1]], p[el[2]]
        b1, b2, b3 = np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])
        coeff = np.array([np.linalg.solve(B, b1), np.linalg.solve(B, b2), np.linalg.solve(B, b3)])

        C =np.array(([1,nu,0],[nu,1,0],[0,0,(1-nu)/2]))*E/(1-nu**2)

        # Create elemental matrix Ak and vector F using Gaussian quadrature  + assembly
        func = lambda x, y, c: x * 0 + y * 0 + c
        for i in range(len(el)):
            for j in range(len(el)):
                t1=np.array([np.array([coeff[i,1],0,coeff[i,2]]),np.array([0,coeff[i,2],coeff[i,1]])])
                t2=[np.array([coeff[j,1],0,coeff[j,2]]),np.array([0,coeff[j,2],coeff[j,1]])]
                for d1 in range(2):
                    for d2 in range(2):
                        w = t1[d1]@C@t2[d2]
                        A[2*el[i]+d1,2*el[j]+d2] += quadrature2D(p[el[0]], p[el[1]], p[el[2]], Nq, func, w)
            for d in range(2):
                F[2*el[i]+d] += quadrature2D(p1, p2, p3, Nq, lin, coeff[i], f,d)
    return A, F, edge, p, tri

def homogeneousDirichlet(N, Nq, f,nu,E):
    """
    Solving the poisson problem in 2D with homogeneous Dirichlet BCs.
    N is the number of nodes in triangulation, Nq is the number of
    integration points in the gaussian quadrature. f is the rhs of the eq.
    Returns the solution u and a list of coordinates of nodes p.
    """
    A, F, edge, p, tri = createAandF(f, N, Nq,nu,E)
    nodes = np.unique(edge)
    nodes = nodes -1
    epsilon = 1e-16
    for d in range(2):

        F[2*nodes+d]=0
        A[2*nodes+d, 2*nodes+d] = 1 / epsilon
    A=sparse.csr_matrix(A)
    u = spsolve(A, F)
    return u, p, tri