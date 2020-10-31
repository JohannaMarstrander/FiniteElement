# -*- coding: utf-8 -*-
import unittest

import numpy as np
from Part2_code.stress_recovery import StressRecovery
from Part2_code.test2 import homogeneousDirichlet

E=1
nu=0.25

def f(x,y,pos):
    if pos==0:
        return E/(1-nu**2) * (-2*y**2 - x**2 + nu* x**2 -2*nu*x*y - 2*x*y + 3 -nu)
    else:
        return E/(1-nu**2) * (-2*x**2 - y**2 + nu* x**2 -2*nu*x*y - 2*x*y + 3 -nu)


def u(x,y):
    return (x**2-1)*(y**2-1)

class TestHomogeneousDirichlet(unittest.TestCase):

    def test_compare_analytic(self):
        N = 50
        
        #Find numerical solution
        u_num, p, tri=homogeneousDirichlet(N,4,f,nu,E)
        u1_num=u_num[::2]
        u2_num=u_num[1::2]
        
        #Find exact solution (equal components)
        u_ex=u(p[:, 0], p[:, 1])
        
        #Compare
        max_error = np.max((np.max(np.abs(u1_num-u_ex)), np.max(np.abs(u2_num-u_ex))))
        print("Max error is:", max_error, " for N=",N)
        self.assertAlmostEqual(max_error, 0, delta=1e1/N)
        
        plot(p[:, 0], p[:, 1], u1_num, "Numerical Solution, N = "+str(N), set_axis = True)
        plot(p[:, 0], p[:, 1], u_ex, "Exact Solution", set_axis = True)
        plot(p[:, 0], p[:, 1], u1_num - u_ex, "Error, N = "+str(N))
        
    def test_convergence(self):
        return 0

def e(x,y):
    return 2*x*(y**2 - 1)
    
class TestStressRecovery(unittest.TestCase):
    
    def test_stress_recovery(self):
        N = 20
        #Find numerical solution
        u, p, tri=homogeneousDirichlet(N,4,f,nu,E)
        
        S, p = StressRecovery(u,p,tri,nu, E)
        e_ex = e(p[:,0], p[:,1])
        
        C =np.array(([1,nu,0],[nu,1,0],[0,0,(1-nu)/2]))*E/(1-nu**2)
        C_inv = np.linalg.inv(C)
        
        A = C_inv@S
        max_error = np.max(np.abs(e_ex-A[0]))
        self.assertAlmostEqual(max_error, 0, delta=1e1/N)
        
        plot(p[:, 0], p[:, 1], A[0], "sigma_xx, N = ")
        plot(p[:, 0], p[:, 1], e_ex, "sigma_yy, N = ")
        plot(p[:, 0], p[:, 1], S[0], "sigma_xy, N = ")
        
        #Hva skal egentlig testes her?

        


if __name__ == '__main__':
    unittest.main()
