# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 15:15:50 2020

@author: johan
"""

"""
Function for creating A and F with Neumann boundary conditions as given in task 3.
Uses base function from poisson2D.
"""

from poisson2D import createAandF, lin
from Quadrature import quadrature1D
import numpy as np
from plot import plot

def neumann(N, Nq, f, g):
    """
    Solving the poisson problem in 2D with Neumann and Dirichlet BCs as given in task 3.
    N is the number of nodes in triangulation, Nq is the number of integration points 
    in the gaussian quadrature. f is the rhs of the eq and g is the derivative at the 
    neumann boundary. Returns the solution u and a list of coordinates of nodes p.
    """
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



