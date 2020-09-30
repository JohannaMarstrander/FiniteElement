from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot(x, y, z):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(x, y, z, linewidth=0.2, antialiased=True)
    plt.show()
