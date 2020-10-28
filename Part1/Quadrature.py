"""
Module with functions for Gaussian Quadrature in 1D and 2D, including 
line integral in 2D.
"""
import numpy as np 

def quadrature1D(a, b, Nq ,g, *args):
    """
    Takes in two points and uses Gaussian quadrature to approximate the integral 
    of the function g over the straight line between them. If a, b are numbers, 
    a normal integral in R is assumed. If a, b are arrays of length 2, the line
    integral over the straight line between them in R^2 is computed.
    Nq is the number of integration points. 
    """
    zq,pq = np.polynomial.legendre.leggauss(Nq)
    if type(a) == list or type(a) == np.ndarray:
        #assumes len(a) == len(b) == 2
        a, b = np.array(a), np.array(b)
        dist = np.linalg.norm(a-b)
        xq = 0.5*(b[0]-a[0])*zq+0.5*(b[0]+a[0])
        yq = 0.5*(b[1]-a[1])*zq+0.5*(b[1]+a[1])
        g_gauss = 0.5*dist*np.dot(g(xq, yq, *args),pq)
    else:
        xq = 0.5*(b-a)*zq+0.5*(b+a)
        g_gauss = 0.5*(b-a)*np.dot(g(xq),pq)
    return g_gauss


def quadrature2D(p1,p2,p3,Nq,g,*args):
    """
    Uses Gaussian quadrature to approximate the definite integral of g over the
    triangle defined by the points p1, p2, p3. 
    Nq is the number of integration points. 
    """
    n=[100,0,100,1,2] #maps from Nq, 100 to mark not a valid Nq
    assert(n[Nq]!=100), "Not a valid number of integration points. Must be 1,3 or 4."
    pq=[[1],[1/3,1/3,1/3],[-9/16,25/48,25/48,25/48]]
    zeta=[[1/3,1/3,1/3],[[1/2,1/2,0],[1/2,0,1/2],[0,1/2,1/2]],[[1/3,1/3,1/3],[3/5,1/5,1/5],[1/5,3/5,1/5],[1/5,1/5,3/5]]]
    area=0.5*abs(np.linalg.norm(np.cross(p1-p2,p3-p2)))
    xq=np.dot(zeta[n[Nq]],np.array([p1,p2,p3]))
    if Nq == 1:
        return np.dot(pq[n[Nq]],g(xq[0],xq[1],*args))
    g_gauss=area*np.dot(pq[n[Nq]],g(xq[:,0],xq[:,1],*args))
    return g_gauss
