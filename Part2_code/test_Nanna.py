from Part2_code.getplate import getPlate
from matplotlib import pyplot as plt
import numpy as np
from Part1.Quadrature import quadrature2D
from Part1.plot import plot
import pickle


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
        f is the rhs of the eq, N^2 is number of nodes, Nq number of integration points in
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
    nodes = np.unique(edge)
    nodes = nodes -1
    epsilon = 1e-16
    for d in range(2):

        F[2*nodes+d]=0
        A[2*nodes+d, 2*nodes+d] = 1 / epsilon
    u = np.linalg.solve(A, F)
    return u, p, tri


#u,p=homogeneousDirichlet(4,4,f,0.25,1)

#a=np.load(outfile)

#with open('test.npy', 'wb') as f:
#    np.save(f, u)



#u1_num=u[::2]
#u2_num=u[1::2]

#print(u)

def u(x,y):
    return (x**2-1)*(y**2-1)

#u1_ex=u(p[:, 0], p[:, 1])
#
#
#plot(p[:, 0], p[:, 1], u1_num, "Numerical Solution, N = ", set_axis = True)
#plot(p[:, 0], p[:, 1], u1_ex, "Exact Solution", set_axis = True)
#plot(p[:, 0], p[:, 1], u1_num - u1_ex, "Error, N = ")
#
#plot(p[:, 0], p[:, 1], u2_num, "2Numerical Solution, N = ", set_axis = True)
#plot(p[:, 0], p[:, 1], u2_num - u1_ex, "2Error, N = ")
#
#PlotMesh(10)
#plt.show()

def error():
    U=[]
    n=2**7
    rel_error=[]
    conv=[]
    h=[]
    u,p, tri = homogeneousDirichlet(n+1, 4, f, 0.25, 1)
    ux,uy= u[::2],u[1::2]
    U.append(u)
    for i in range(1,6):
        #print(i)
        N=2**i+1
        t = int(n/2**(i))
        #print(t)

        a = np.linspace(0, n, n+1)
        k = np.array([n * a[::t] + j for j in a[::t]]).flatten()
        k=k.astype(int)
        #print(k)
        ux_k=ux[k]
        #u_best = np.array([ux[::t],uy[::t]]) #making the comparison easiest
        #print(u_best)
        u, p, tri = homogeneousDirichlet(N, 4, f, 0.25, 1)
        ux_new= u[::2]
        #print(ux_k)
        #print(ux_new)
        error=  abs(ux_new - ux_k)/np.linalg.norm(ux,ord=np.inf)
        rel_error.append( error)
        conv.append(np.linalg.norm(error))
        print(max(error))
        h.append(1/(2**i))
    return rel_error, conv,h

#rel_error,conv,h=error()

#print(rel_error)
#plt.figure()
#plt.loglog(h,conv)
#plt.show()
#order = np.polyfit(np.log(h), np.log(conv), 1)[0]
#print("order", order)



#p,tri,edge=getPlate(10)
#print(p,tri,edge)