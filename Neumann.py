# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 15:15:50 2020

@author: johan
"""

from poisson2D import createAandF, lin
from Task1 import quadrature2D
import numpy as np

def neumann(N, Nq, f, g):
    epsilon = 1e-16
    print("hallois")
    A,F,edge,p, tri = createAandF(f,N, Nq)
    nodes = np.unique(edge)

    #Dirichlet
    nodes_dirichlet = nodes[p[nodes,1]<=0]
    F[nodes_dirichlet] = 0
    A[nodes_dirichlet, nodes_dirichlet] = 1/epsilon
    
    #Neumann
    nodes_neumann = nodes[p[nodes,1]>0]
    for node in nodes_neumann:
        for el in tri[np.any(tri == node, axis = 1), :]:
            #Find coefficients
            C = np.hstack((np.array([[1], [1], [1]]), p[el]))
            index = np.argwhere(el==node)
            b = np.zeros(3)
            b[index] = 1
            coeff = np.linalg.solve(C, b)
            
            #Modify F
            F[node] += quadrature2D(p[el[0]], p[el[1]], p[el[2]], Nq, lin, coeff, g)

    u = np.linalg.solve(A, F)
    return u, p

def f(x, y):
    return x + y

neumann(5,4,f,f)

