from Part2_code.getplate import getPlate
from matplotlib import pyplot as plt
import numpy as np
from Part2_code.linearElasticity2D import  homogeneousDirichlet
from Part1.plot import plot



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

E=1
nu=0.25
def f(x,y,pos):
    if pos==0:
        return E/(1-nu**2) * (-2*y**2 - x**2 + nu* x**2 -2*nu*x*y - 2*x*y + 3 -nu)
    else:
        return E/(1-nu**2) * (-2*x**2 - y**2 + nu* x**2 -2*nu*x*y - 2*x*y + 3 -nu)


def u(x,y):
    return (x**2-1)*(y**2-1)


def error():
    n=2**6
    rel_error=[]
    conv=[]
    h=[]
    u,p, tri = homogeneousDirichlet(n+1, 4, f, 0.25, 1)
    ux,uy= u[::2],u[1::2]
    a = np.linspace(0, n, n + 1)
    for i in range(1,5):
        N=2**i+1
        t = int(n/(2**(i)))
        k = np.array([(n+1) * a[::t] + j for j in a[::t]]).flatten()
        k=k.astype(int)
        k=np.sort(k)
        ux_k,uy_k = ux[k],uy[k]
        u, p,tri = homogeneousDirichlet(N, 4, f, 0.25, 1)
        u_num= np.hstack((u[::2] ,u[1::2]))
        u_k=np.hstack((ux_k,uy_k))
        error=  abs(u_k - u_num)/np.linalg.norm(ux,ord=np.inf)
        rel_error.append(error)
        conv.append(np.linalg.norm(u_num - u_k))
        h.append(1/(2**i))
    return rel_error, conv,h

rel_error,conv,h=error()

plt.figure()
plt.loglog(h,conv)
plt.show()
order = np.polyfit(np.log(h), np.log(conv), 1)[0]
print("order", order)


