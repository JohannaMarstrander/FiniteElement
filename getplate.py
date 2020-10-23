# Description:
#   Generate a mesh triangulation of the reference square (-1,1)^2.
#
# Arguments:
#   N       Number of nodes in each spatial direction (N^2 total nodes).
#
# Returns:
#   p		Nodal points, (x,y)-coordinates for point i given in row i.
#   tri   	Elements. Index to the three corners of element i given in row i.
#   edge  	Index list of all nodal points on the outer edge (r=1).
#
#   Author: Kjetil A. Johannessen, Abdullah Abdulhaque
#   Last edit: October 2019


import numpy as np
import scipy.spatial as spsa


def getPlate(N):
    # Defining auxiliary variables.
    L = np.linspace(-1,1,N)
    Y,X = np.meshgrid(L,L)
    x = np.ravel(np.transpose(X))
    y = np.ravel(np.transpose(Y))

    # Generating nodal points.
    p = np.zeros((N**2,2))
    for i in range(0,N**2):
        p[i,0] = x[i]
        p[i,1] = y[i]

    # Generating elements.
    mesh = spsa.Delaunay(p)
    tri = mesh.simplices

    # Generating nodal points on outer edge.
    south = np.array([np.arange(1,N),np.arange(2,N+1)])
    east = np.array([np.arange(N,N**2-N+1,N),np.arange(2*N,N**2+1,N)])
    north = np.array([np.arange(N**2,N**2-N+1,-1),np.arange(N**2-1,N**2-N,-1)])
    west = np.array([np.arange(N**2-N+1,N-1,-N),np.arange(N**2-2*N+1,0,-N)])
    L1 = np.shape(south)[1]
    L2 = np.shape(east)[1]
    L3 = np.shape(west)[1]
    L4 = np.shape(north)[1]
    edge = np.zeros((L1+L2+L3+L4,2),dtype=np.int)
    for i in range(0,L1):
        edge[i,0] = south[0,i]
        edge[i,1] = south[1,i]
    for i in range(L1,L1+L2):
        edge[i,0] = east[0,i-L1]
        edge[i,1] = east[1,i-L1]
    for i in range(L1+L2,L1+L2+L3):
        edge[i,0] = north[0,i-L1-L2]
        edge[i,1] = north[1,i-L1-L2]
    for i in range(L1+L2+L3,L1+L2+L3+L4):
        edge[i,0] = west[0,i-L1-L2-L3]
        edge[i,1] = west[1,i-L1-L2-L3]

    return p,tri,edge