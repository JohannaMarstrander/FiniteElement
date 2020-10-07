# -*- coding: utf-8 -*-
"""
Solving the Poisson problem with both dirichlet and neumann BCs, as well as
testing quadrature functions, plotting meshes and answering subtasks in project.
"""
import numpy as np
from Quadrature import quadrature1D, quadrature2D
from plot import PlotMesh, plot
from poisson2D import createAandF
from homDirichlet import homogeneousDirichlet
from Neumann import neumann 

"""Testing quadrature1D (1a)"""
def g1(x):
    return np.exp(x)

exact = np.exp(2)-np.exp(1)
err_1D = [0]*4
for N in [1,2,3,4]:
    integral = quadrature1D(1,2,N,g1)
    err_1D[N-1] = np.abs(integral- exact)
#print(err_1D)

"""Testing quadrature2D(1b)"""

def g2(x,y):
    return np.log(x+y)

exact = 1.16542
err_2D = [0]*4
for N in [1,3,4]:
    integral = quadrature2D(np.array([1,0]),np.array([3,1]),np.array([3,2]),N,g2)
    err_2D[N-1] = abs(integral -exact)
#print(err_2D)

"""Task 2"""
def f(x, y):
    return -8 * np.pi * np.cos(2 * np.pi * (x ** 2 + y ** 2)) + 16 * np.pi ** 2 * (x ** 2 + y ** 2) * np.sin(2 * np.pi * (x ** 2 + y ** 2))

def u(x, y):
    return np.sin(2 * np.pi * (x ** 2 + y ** 2))

"""Plotting 3 meshes (task 2d)"""
    
#PlotMesh(15)
#PlotMesh(50)
#PlotMesh(100)

"""Verifying that A i singular(task 2e)"""
A, F, edge, p, tri = createAandF(f, 100, 4)

#Checking condition number to see if A is singular up to machine precision
if np.linalg.cond(A) > 1/np.finfo(A.dtype).eps:
    print("The matrix A i singular")


"""Solve the system with homogeneous dirichlet BCs and comparing solution(task 2g/f)"""
N = 512
u_num, p = homogeneousDirichlet(N, 4, f)
u_ex = u(p[:, 0], p[:, 1])

plot(p[:, 0], p[:, 1], u_num, "Numerical Solution, N = "+str(N), set_axis = True)
plot(p[:, 0], p[:, 1], u_ex, "Exact Solution", set_axis = True)
plot(p[:, 0], p[:, 1], u_num - u_ex, "Error, N = "+str(N))


"""Solve the system with Neumann BCs as given in task 3 (task 3)"""
def g(x, y):
    return 4*np.pi * np.sqrt(x*x + y*y) * np.cos(2*np.pi * (x*x + y*y))

u_num, p = neumann(512, 4, f, g)
u_ex = u(p[:, 0], p[:, 1])

plot(p[:, 0], p[:, 1], u_num, "Numerical Solution, N = "+str(N), set_axis = True)
plot(p[:, 0], p[:, 1], u_ex, "Exact Solution", set_axis = True)
plot(p[:, 0], p[:, 1], u_num - u_ex, "Error, N = "+str(N))