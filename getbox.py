# Description:
#   Generate a mesh triangulation of the unit box (0,1)^3.
#
# Arguments:
#   N       Number of nodes in each spatial direction (N^3 total nodes).
#
# Returns:
#   p       Nodal points. (x,y,z)-coordinates for point i given in row i.
#   tet     Elements. Index to the four corners of element i given in row i.
#   edge    Index list of all nodal points on the outer edge (r=1)
#
#   Author: Kjetil A. Johannessen, Abdullah Abdulhaque
#   Last edit: October 2019


import numpy as np
import scipy.spatial as spsa


def getBox(N):
    # Defining auxiliary variables.
    L = np.linspace(0,1,N)
    Y,X,Z = np.meshgrid(L,L,L)
    x = np.ravel(np.transpose(X))
    y = np.ravel(np.transpose(Y))
    z = np.ravel(np.transpose(Z))

    # Generating the nodal points.
    p = np.zeros((N**3,3))
    for i in range(0,N**3):
        p[i,0] = x[i]
        p[i,1] = y[i]
        p[i,2] = z[i]

    # Generating the elements.
    mesh = spsa.Delaunay(p)
    tet = NodalPoints(p,mesh)

    # Generating the boundary elements.
    edge = freeBoundary(mesh)

    return p,tet,edge


def NodalPoints(p,mesh):
    tet_temp = mesh.simplices
    tet = []
    for t in tet_temp:
        x1 = p[t[0],:]
        x2 = p[t[1],:]
        x3 = p[t[2],:]
        x4 = p[t[3],:]
        v1 = x2-x1
        v2 = x3-x1
        v3 = x4-x1
        V6 = np.abs(np.dot(np.cross(v1,v2),v3))
        if V6 >= 10**-13:
            tet += [t]
    return np.array(tet)


def freeBoundary(mesh):
    # Auxiliary function for generating boundary nodes.
    edge = []
    for ind, neigh in zip(mesh.simplices, mesh.neighbors):
        for j in range(4):
            if neigh[j] == -1:
                edge += [[ind[j-1],ind[j-2],ind[j-3]]]
    return np.array(edge)
