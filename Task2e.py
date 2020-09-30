# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 10:35:29 2020

@author: johan
"""
from getdisc import GetDisc
import numpy as np
from Task1 import quadrature2D
#from createF import createF

def lin(x,y,c,g):
    return (c[0]+x*c[1]+y*c[2])*g(x,y)


def createF(g,N,Nq):
    p, tri, edge = GetDisc(N)
    F = np.zeros(N)
    T = np.ones((3, 3))
    for el in tri:
        p1,p2,p3=p[el[0]],p[el[1]],p[el[2]]
        T[:,1:]=p1,p2,p3
        C = np.linalg.solve(T, alpha)
        for i in range(0,2):
            F[el[i]]+=quadrature2D(p1,p2,p3,Nq,lin,C[i],g)
    return F

def createAandF(g,N, Nq):
    #Returns A as well as list of corners of edge lines for incorporating BCs
    p, tri, edge = GetDisc(N)
    A = np.zeros((len(p), len(p)))
    F = np.zeros(N)

    
    for el in tri: #for each element
        #Find coefficients of H_alpha^k
        C = np.hstack((np.array([[1],[1],[1]]), p[el]))
        p1, p2, p3 = p[el[0]], p[el[1]], p[el[2]]
        b1, b2, b3 = np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1])
        coeff = np.array([np.linalg.solve(C, b1), np.linalg.solve(C, b2), np.linalg.solve(C, b3)])

        #Create elemental matrix Ak using Gaussian quadrature  + assembly
        func = lambda x,y,c: x*0 + y*0 + c
        for i in range(len(el)):
            for j in range(len(el)):
                c = np.dot(coeff[i,1:], coeff[j,1:])
                A[el[i],el[j]] += quadrature2D(p[el[0]],p[el[1]],p[el[2]],Nq,func, c)
            F[el[i]] += quadrature2D(p1, p2, p3, Nq, lin, coeff[i], g)
    return A,F, edge,p

#A, edge = createA(5,4)

#print("A:",A)

def homogeneousDirichlet(N, Nq,f):
    #F = createF(f,N,Nq)
    A,F, edge,p = createAandF(f,N, Nq)
    nodes = np.unique(edge)
    F[nodes]=0
    epsilon = 1e-16
    A[nodes, nodes] = 1/epsilon
    u=np.linalg.solve(A,F)
    return u,p

def f(x,y):
    return -8*np.pi*np.cos(2*np.pi*(x**2+y**2))+16*np.pi**2*(x**2+y**2)*np.sin(2*np.pi*(x**2+y**2))

def u(x,y):
    return np.sin(2*np.pi*(x**2+y**2))



u_num,p=homogeneousDirichlet(100,4,f)

u_ex=u(p[:,0],p[:,1])
print(u_num)
print(u_ex)

print(u_num-u_ex)
