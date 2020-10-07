This is a project in TMA4220 Numerical Solution of Partial Differential Equations Using Element Methods,where we build
up a code base for later solving finite element problems. We implement Gaussian quadrature, and program for building
a stiffness matrix and a right hand side vector for solving the poisson problem in 2D. Then we solve an example problem
including both Dirichlet and Neumann boundary conditions.

Running test.py all the tests required in the project is executed

Files:

test.py: Solving the Poisson problem with both dirichlet and neumann BCs, as well as
testing quadrature functions, plotting meshes and answering subtasks in project.
getdisc.py: Generate a mesh triangulation of the unit disc.
plot.py: Functions for plotting the mesh and plotting solutions.
Quadrature.py: Module with functions for Gaussian Quadrature in 1D and 2D, including
line integral in 2D.
poisson2D: Base function for creating A and F without considering boundary conditions.
Neumann.py: Function for creating A and F with Neumann boundary conditions.
homDirichlet.py: Function for creating A and F with homogeneous Dirichlet boundary conditions

