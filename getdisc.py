# Description:
#   Generate a mesh triangulation of the unit disc.
#
# Arguments:
#   N       Number of nodes in the mesh.
#
# Returns:
#   p		Nodal points, (x,y)-coordinates for point i given in row i.
#   tri   	Elements. Index to the three corners of element i given in row i.
#   edge  	Edge lines. Index list to the two corners of edge line i given in row i.
#
#   Author: Kjetil A. Johannessen, Abdullah Abdulhaque
#   Last edit: 09-10-2019


import numpy as np
import scipy.spatial as spsa


def GetDisc(N):
	# Controlling the input.
	if N < 4:
		print("Error. N >= 4 reguired for input.")
		return

	# Defining auxiliary variables.
	M,r,alpha,theta = CircleData(N)

	# Generating the nodal points.
	p = NodalPoints(M,N,alpha,theta,r)

	# Generating the elements.
	mesh = spsa.Delaunay(p)
	tri = mesh.simplices

	# Generating the boundary elements.
	edge = FreeBoundary(N,alpha)

	return p,tri,edge

def NodalPoints(M,N,alpha,theta,r):
	# Auxiliary function for generating nodal points.
	p = np.zeros((N,2))
	k = 1
	for i in range(1,M+1):
		t = theta[i]
		for j in range(0,alpha[i]):
			p[k,:] = [np.cos(t)*r[i],np.sin(t)*r[i]]
			t += 2*np.pi/alpha[i]
			k += 1

	return p


def FreeBoundary(N,alpha):
	# Auxiliary function for generating boundary nodes.
	E = np.arange(N-alpha[-1]+1,N+1)
	edge = np.zeros((len(E),2),dtype=np.int)
	for i in range(0,len(E)):
		edge[i,:] = [E[i],E[i]+1]
	edge[-1,-1] = N-alpha[-1]+1
	edge -= 1

	return edge


def CircleData(N):
	# Number of outward circles,excluding the origin.
	M = np.int(np.floor(np.sqrt(N/np.pi)))

	# Radius of the different circles.
	r = np.linspace(0,1,M+1)

	# Number of DOF in each circle.
	alpha_temp = np.floor((2*np.pi*M)*r)
	alpha = np.zeros(len(alpha_temp),dtype=np.int)
	for i in range(0,len(alpha_temp)):
		alpha[i] = np.int(alpha_temp[i])

	# Fine-tuning to get the right amount of DOF.
	alpha[0] = 1
	i = 1
	while sum(alpha) > N:
		if alpha[i] > 0:
			alpha[i] -= 1
		i += 1
		if sum(alpha[1:M]) == 0:
			i = M
		elif i > M:
			i = 1
	while sum(alpha) < N:
		alpha[-1] += 1

	# Creating the starting angle.
	theta = np.pi/alpha
	theta[0:len(alpha):2] = 0

	return M,r,alpha,theta
