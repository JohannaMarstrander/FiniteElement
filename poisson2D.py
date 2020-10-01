# -*- coding: utf-8 -*-


from getdisc import GetDisc
import numpy as np
from Quadrature import quadrature2D


def lin(x, y, c, g):
    "helping to create the F vecotr"
    return (c[0] + x * c[1] + y * c[2]) * g(x, y)

def createAandF(f, N, Nq):
    # Returns A as well as list of corners of edge lines for incorporating BCs
    """Returns A,F ,a list of corners of edge lines and a list of points"""
    p, tri, edge = GetDisc(N)
    A = np.zeros((len(p), len(p)))
    F = np.zeros(N)

    for el in tri:  # for each element
        # Find coefficients of H_alpha^k
        C = np.hstack((np.array([[1], [1], [1]]), p[el]))
        p1, p2, p3 = p[el[0]], p[el[1]], p[el[2]]
        b1, b2, b3 = np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])
        coeff = np.array([np.linalg.solve(C, b1), np.linalg.solve(C, b2), np.linalg.solve(C, b3)])

        # Create elemental matrix Ak and vector F using Gaussian quadrature  + assembly
        func = lambda x, y, c: x * 0 + y * 0 + c
        for i in range(len(el)):
            for j in range(len(el)):
                c = np.dot(coeff[i, 1:], coeff[j, 1:])
                A[el[i], el[j]] += quadrature2D(p[el[0]], p[el[1]], p[el[2]], Nq, func, c)
            F[el[i]] += quadrature2D(p1, p2, p3, Nq, lin, coeff[i], f)
    return A, F, edge, p, tri
