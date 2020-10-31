# -*- coding: utf-8 -*-
import numpy as np
from Part2_code.test2 import homogeneousDirichlet
from Part1.plot import plot

E=1
nu=0.25
def f(x,y,pos):
    if pos==0:
        return E/(1-nu**2) * (-2*y**2 - x**2 + nu* x**2 -2*nu*x*y - 2*x*y + 3 -nu)
    else:
        return E/(1-nu**2) * (-2*x**2 - y**2 + nu* x**2 -2*nu*x*y - 2*x*y + 3 -nu)
    
u,p, tri=homogeneousDirichlet(100,4,f,nu,E)

def StressRecovery(U, p, tri, nu, E):
    #Finding average stress per element(assuming const)
    e = np.zeros((len(tri),3))
    S = np.zeros((len(p), 4))
    
    for n in range(len(tri)):
        el = tri[n]
    
        # Find coefficients of u_1, u_2
        A = np.hstack((np.array([[1], [1], [1]]), p[el]))
        u1_x, u2_x, u3_x = U[2*el]
        u1_y, u2_y, u3_y = U[2*el + 1]
        b1, b2 = np.array([u1_x, u2_x, u3_x]), np.array([u1_y, u2_y, u3_y])
        coeff = np.array([np.linalg.solve(A, b1), np.linalg.solve(A, b2)])
        
        #Finding strain
        e[n,::2] += coeff[0,1:]
        e[n,1:] += coeff[1,1:]
        #Finding stress
        C =np.array(([1,nu,0],[nu,1,0],[0,0,(1-nu)/2]))*E/(1-nu**2)
        e[n] = C@e[n]
        
        #Adding contribution to nodes
        for i in el:
            S[i,1:] += e[n]
            S[i,0] += 1.
        
    return (S[:,1:]/S[:,0][:,None]).T, p
    
S, p = StressRecovery(u,p,tri,nu, E)

u1_num=u[::2]
u2_num=u[1::2]

plot(p[:, 0], p[:, 1], u1_num, "Numerical Solution, N = ")
plot(p[:, 0], p[:, 1], u2_num, "2Numerical Solution, N = ")
plot(p[:, 0], p[:, 1], S[0], "Stress 1, N = ")
plot(p[:, 0], p[:, 1], S[1], "Stress 2, N = ")
plot(p[:, 0], p[:, 1], S[2], "Stress 3, N = ")
