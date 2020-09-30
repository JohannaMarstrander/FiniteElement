import numpy as np 


def quadrature1D(a, b, Nq ,g):
    zq,pq = np.polynomial.legendre.leggauss(Nq)
    xq = 0.5*(b-a)*zq+0.5*(b+a)
    g_gauss = 0.5*(b-a)*np.dot(g(xq),pq)
    return g_gauss

def g(x):
    return np.exp(x)


integral1=quadrature1D(0,2,5,g)

#print(integral1)


def quadrature2D(p1,p2,p3,Nq,g,*args):
    n=[100,0,100,1,2] #maps from Nq, 100 to mark not a valid Nq
    pq=[[1],[1/3,1/3,1/3],[-9/16,25/48,25/48,25/48]]
    zeta=[[1/3,1/3,1/3],[[1/2,1/2,0],[1/2,0,1/2],[0,1/2,1/2]],[[1/3,1/3,1/3],[3/5,1/5,1/5],[1/5,3/5,1/5],[1/5,1/5,3/5]]]
    area=0.5*abs(np.linalg.norm(np.cross(p1-p2,p3-p2)))
    xq=np.dot(zeta[n[Nq]],np.array([p1,p2,p3]))
    if Nq == 1:
        return np.dot(pq[n[Nq]],g(xq[0],xq[1],*args))
    g_gauss=area*np.dot(pq[n[Nq]],g(xq[:,0],xq[:,1],*args))
    return g_gauss

           
def g2(x,y):
    return np.log(x+y)
    #return x*0+y*0+1

integral2=quadrature2D(np.array([1,0]),np.array([3,1]),np.array([3,2]),4,g2)
#print(integral2)