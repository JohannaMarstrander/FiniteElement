from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Part1.getdisc import GetDisc
from matplotlib import pyplot as plt

def PlotMesh(N):
    """Plotting mesh covering the unit disc with N nodes"""
    p, tri, edge = GetDisc(N)
    
    fig, ax = plt.subplots(figsize = (3,3))
    for el in tri:
        ax.plot(p[el, 0], p[el, 1],  "ro-", color = "black")
        ax.plot(p[el[[2,0]], 0], p[el[[2,0]], 1], "ro-",color = "black")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(str(N)+" nodes")
    return ax

def plot(x, y, z, title, set_axis = False):
    """Plotting solution on triangulation-mesh.
    x, y are node coordinates, z are values."""
    fig = plt.figure(figsize = (4,4))
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(x, y, z, linewidth=0.2, antialiased=True)
    if set_axis:
        ax.set_zlim((-0.75,1.0))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    plt.show()
