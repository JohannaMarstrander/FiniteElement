# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 10:35:29 2020

@author: johan
"""
from getdisc import GetDisc
import numpy as np
from Task1 import quadrature2D

def createA(N, Nq):
    #Returns A as well as list of corners of edge lines for incorporating BCs
    p, tri, edge = GetDisc(N)
    A = np.zeros((len(p), len(p)))
    
    for el in tri: #for each element
        #Find coefficients of H_alpha^k
        C = np.hstack((np.array([[1],[1],[1]]), p[el]))
        b1, b2, b3 = np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1])
        coeff = np.array([np.linalg.solve(C, b1), np.linalg.solve(C, b2), np.linalg.solve(C, b3)])

        #Create elemental matrix Ak using Gaussian quadrature  + assembly
        func = lambda x,y,c: x*0 + y*0 + c
        for i in range(len(el)):
            for j in range(len(el)):
                c = np.dot(coeff[i,1:], coeff[j,1:])
                A[el[i],el[j]] += quadrature2D(p[el[0]],p[el[1]],p[el[2]],Nq,func, c)
    return A, edge

A, edge = createA(5,4)

print("A:",A)

def homogeneousDirichlet(N, Nq):
    A, edge = createA(N, Nq)
    nodes = np.unique(edge)
    epsilon = 1e-10
    A[nodes, nodes] = 1/epsilon
    return A

print("A_eps:", homogeneousDirichlet(5,4))