import numpy as np
from getdisc import GetDisc
from Task1 import quadrature2D

#   p		Nodal points, (x,y)-coordinates for point i given in row i.
#   tri   	Elements. Index to the three corners of element i given in row i.



alpha=np.array([[1,0,0],[0,1,0],[0,0,1]])

def g(x,y):
    return 1

def f(x,y,c,g):
    return (c[0]+x*c[1]+y*c[2])*g(x,y)

def createF(g,N):
    p, tri, edge = GetDisc(N)
    F = np.zeros(N)
    T = np.ones((3, 3))
    for el in tri:
        p1,p2,p3=p[el[0]],p[el[1]],p[el[2]]
        T[:,1:]=p1,p2,p3
        C = np.linalg.solve(T, alpha)
        for i in range(0,2):
            F[el[i]]+=quadrature2D(p1,p2,p3,4,f,C[i],g)
    return F

print(createF(g,10))
