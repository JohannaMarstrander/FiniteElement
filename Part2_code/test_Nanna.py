from Part2_code.getplate import getPlate
from matplotlib import pyplot as plt
import numpy as np
from Part2_code.linearElasticity2D import  homogeneousDirichlet
from Part1.plot import plot
import time




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
        return E/(1-nu**2) * (-2*x**2 - y**2 + nu* y**2 -2*nu*x*y - 2*x*y + 3 -nu)


def u(x,y):
    return (x**2-1)*(y**2-1)

#N=100
#u_num,p,tri=homogeneousDirichlet(N,4,f,nu,E)
#u1_num=u_num[::2]
#u2_num=u_num[1::2]
#u_ex=u(p[:, 0], p[:, 1])
#plot(p[:, 0], p[:, 1], u1_num, "Numerical Solution u1, N = " + str(N**2),"u1", set_axis=True)
#plot(p[:, 0], p[:, 1], u2_num, "Numerical Solution u2, N = " + str(N**2),"u2" ,set_axis=True)
#plot(p[:, 0], p[:, 1], u_ex, "Exact Solution ","exact", set_axis=True)
#plot(p[:, 0], p[:, 1], u_num - u_ex, "Error, N = " + str(N))



def error(nu,E,f,highest):
    n=2**highest
    rel_error=[]
    conv=[]
    h=[]
    u,p, tri = homogeneousDirichlet(n+1, 4, f, nu, E)
    ux,uy= u[::2],u[1::2]
    a = np.linspace(0, n, n + 1)

    for i in range(1,highest):
        N=2**i+1
        t = int(n/(2**(i)))
        k = np.array([(n+1) * a[::t] + j for j in a[::t]]).flatten()
        k=k.astype(int)
        k=np.sort(k)
        ux_k,uy_k = ux[k],uy[k]
        u, p,tri = homogeneousDirichlet(N, 4, f, nu, E)
        conv.append(np.linalg.norm(ux_k - u[::2]))
        h.append(1/(2**i))
    plt.figure()
    plt.loglog(h,conv)
    plt.xlabel("log(h)")
    plt.ylabel("log(error)")
    plt.savefig("conv.pdf")
    plt.show()
    order = np.polyfit(np.log(h), np.log(conv), 1)[0]
    print("order", order)

def check_time(nu,E,f,highest):
    timeList=[]
    h=[]
    for i in range(1,highest):
        N=2**i+1
        start = time.time()
        u, p, tri = homogeneousDirichlet(N, 4, f, nu, E)
        end = time.time()
        timeList.append(end-start)
        h.append(1/(2**i))
    plt.figure()
    plt.plot(h, timeList)
    plt.xlabel("h")
    plt.ylabel("time(seconds)")
    plt.savefig("time.pdf")
    plt.show()


error(nu,E,f,7)
check_time(nu,E,f,7)


