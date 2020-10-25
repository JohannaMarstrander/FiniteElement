from getplate import getPlate
from matplotlib import pyplot as plt
import numpy as np
from Quadrature import quadrature2D
from plot import plot


def PlotMesh(N):
    """Plotting mesh covering the unit disc with N nodes"""
    p, tri, edge = getPlate(N)

    #print(p,tri,edge)
    #print(len(p))
    fig, ax = plt.subplots(figsize=(3, 3))
    edge = edge -1
    for el in tri:
        ax.plot(p[el, 0], p[el, 1], "ro-", color="black")
        ax.plot(p[el[[2, 0]], 0], p[el[[2, 0]], 1], "ro-", color="black")
    for el in edge:
        ax.plot(p[el, 0], p[el, 1], "ro-", color="red")
    #    ax.plot(p[el[[2, 0]], 0], p[el[[2, 0]], 1], "ro-", color="red")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(str(N) + " nodes")
    plt.show()
    return ax

def lin(x, y, c, g,*args):
    "utility function to create the F vecotr"
    return (c[0] + x * c[1] + y * c[2]) * g(x, y,*args)

def createAandF(f, N, Nq,nu,E):
    """Returns A, F,a list of corners of edge lines, a list of points, list of elements.
        f is the rhs of the eq, N is number of nodes, Nq number of integration points in
        gaussian quadrature"""
    p, tri, edge = getPlate(N)
    #print(tri)
    A = np.zeros((2*N**2, 2*N**2))
    F = np.zeros(2*N**2)

    for el in tri:  # for each element
        # Find coefficients of H_alpha^k
        C = np.hstack((np.array([[1], [1], [1]]), p[el]))
        p1, p2, p3 = p[el[0]], p[el[1]], p[el[2]]
        b1, b2, b3 = np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])
        coeff = np.array([np.linalg.solve(C, b1), np.linalg.solve(C, b2), np.linalg.solve(C, b3)])

        C =np.array(([1,nu,0],[nu,1,0],[0,0,(1-nu)/2]))

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
                #c = np.dot(coeff[i, 1:], coeff[j, 1:])
                #A[el[i], el[j]] += quadrature2D(p[el[0]], p[el[1]], p[el[2]], Nq, func, c)
            for d in range(2):
                #print(el[i])
                #print(2*el[i]+d)
                F[2*el[i]+d] += quadrature2D(p1, p2, p3, Nq, lin, coeff[i], f,d)
            #F[el[i]] += quadrature2D(p1, p2, p3, Nq, lin, coeff[i], f)

    A = A*E/(1-nu**2)
    return A, F, edge, p, tri

E=1
nu=0.25
def f(x,y,pos):
    if pos==0:
        return E/(1-nu**2) * (-2*y**2 - x**2 + nu* x**2 -2*nu*x*y - 2*x*y + 3 -nu)
    else:
        return E/(1-nu**2) * (-2*x**2 - y**2 + nu* x**2 -2*nu*x*y - 2*x*y + 3 -nu)


def homogeneousDirichlet(N, Nq, f,nu,E):
    """
    Solving the poisson problem in 2D with homogeneous Dirichlet BCs.
    N is the number of nodes in triangulation, Nq is the number of
    integration points in the gaussian quadrature. f is the rhs of the eq.
    Returns the solution u and a list of coordinates of nodes p.
    """
    A, F, edge, p, tri = createAandF(f, N, Nq,nu,E)
    #print(edge)
    nodes = np.unique(edge)
    #print(nodes)
    nodes = nodes -1
    epsilon = 1e-16
    for d in range(2):

        F[2*nodes+d]=0
        A[2*nodes+d, 2*nodes+d] = 1 / epsilon
    #F[nodes] = 0
    #A[nodes, nodes] = 1 / epsilon
    u = np.linalg.solve(A, F)
    return u, p


u,p=homogeneousDirichlet(60,4,f,0.25,1)

u1_num=u[::2]
u2_num=u[1::2]

#print(u)

def u(x,y):
    return (x**2-1)*(y**2-1)

u1_ex=u(p[:, 0], p[:, 1])
#
#
plot(p[:, 0], p[:, 1], u1_num, "Numerical Solution, N = ", set_axis = True)
plot(p[:, 0], p[:, 1], u1_ex, "Exact Solution", set_axis = True)
plot(p[:, 0], p[:, 1], u1_num - u1_ex, "Error, N = ")
#
plot(p[:, 0], p[:, 1], u2_num, "2Numerical Solution, N = ", set_axis = True)
plot(p[:, 0], p[:, 1], u2_num - u1_ex, "2Error, N = ")
#
#PlotMesh(10)
#plt.show()
