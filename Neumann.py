# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 15:15:50 2020

@author: johan
"""

from poisson2D import createAandF, lin
from Quadrature import quadrature1D
import numpy as np
from plot import plot

def neumann(N, Nq, f, g):
    epsilon = 1e-16
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
            e = [a for a in el if a in nodes and a!=node]
            if len(e)>0:
                e = e[0]
                node_edge = edge[np.any(edge == node, axis = 1)]
                node_edge = node_edge[np.any(node_edge == e, axis = 1)][0]
                if np.argwhere(node_edge == node) == 0:
                    F[node] += quadrature1D(p[node], p[e], Nq, lin, coeff, g)
                else:
                    F[node] += quadrature1D(p[e], p[node], Nq, lin, coeff, g)
    
    u = np.linalg.solve(A, F)
    return u, p

def g(x, y):
    return 4*np.pi * np.sqrt(x*x + y*y) * np.cos(2*np.pi * (x*x + y*y))


def f(x, y):
    return -8 * np.pi * np.cos(2 * np.pi * (x ** 2 + y ** 2)) + 16 * np.pi ** 2 * (x ** 2 + y ** 2) * np.sin(
        2 * np.pi * (x ** 2 + y ** 2))

def u(x, y):
    return np.sin(2 * np.pi * (x ** 2 + y ** 2))

u_num, p = neumann(500, 4, f, g)
u_ex = u(p[:, 0], p[:, 1])

plot(p[:, 0], p[:, 1], u_num)
plot(p[:, 0], p[:, 1], u_ex)
plot(p[:, 0], p[:, 1], u_num - u_ex)

