# FiniteElement
This is a project in TMA4220 Numerical Solution of Partial Differential Equations Using Element Methods. 

In part 1, we build up up a code base for solving finite element problems. We implement Gaussian quadrature,
and program for building a stiffness matrix and a right hand side vector for solving the poisson problem
 in 2D. Then we solve an example problem including both Dirichlet and Neumann boundary conditions.
 
All relevant code and source files are placed in the "/Part1"-directory. Running test.py executed all 
tests required in part 1 of the project.

In part 2, we solve the linear elasticity equation in 2D, find the numerical convergence rate and estimate
time usage and use stress-recovery on the solution for the displacement to calculate the stresses.

All source files and tests for solving the problem are placed in the "/Part2_code" directory. 
Running "test.py" executes all tasks required by the project description. Alternatively, run

python3 -m unittest test.TestClassName.function_name 

from the "/Part2_code" directory to run each test separately. 
