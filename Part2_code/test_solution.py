# -*- coding: utf-8 -*-
import unittest

import numpy as np
from Part2_code.stress_recovery import StressRecovery
from Part2_code.linearElasticity2D import  homogeneousDirichlet
from Part1.plot import plot
from matplotlib import pyplot as plt
import time

E=1
nu=0.25

def f(x,y,pos):
    if pos==0:
        return E/(1-nu**2) * (-2*y**2 - x**2 + nu* x**2 -2*nu*x*y - 2*x*y + 3 -nu)
    else:
        return E/(1-nu**2) * (-2*x**2 - y**2 + nu* y**2 -2*nu*x*y - 2*x*y + 3 -nu)


def u(x,y):
    return (x**2-1)*(y**2-1)

class TestHomogeneousDirichlet(unittest.TestCase):
    
    @unittest.skip
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
        print("Max error of solution is:", max_error, " for N=",N)
        self.assertAlmostEqual(max_error, 0, delta=1/N)
        
        plot(p[:, 0], p[:, 1], u1_num, "Numerical Solution, N = "+str(N), set_axis = True)
        plot(p[:, 0], p[:, 1], u_ex, "Exact Solution", set_axis = True)
        plot(p[:, 0], p[:, 1], u1_num - u_ex, "Error, N = "+str(N))
    
    @unittest.skip
    def test_convergence(self):
        highest = 7 #Comparing num_sol to solution with h = 1/2^highest
        n=2**highest
        rel_error=[]
        conv=[]
        h=[]
        u,p, tri = homogeneousDirichlet(n+1, 4, f, nu, E)
        ux,uy= u[::2],u[1::2]
        a = np.linspace(0, n, n + 1)
    
        for i in range(1,highest-1):
            N=2**i+1
            t = int(n/(2**(i)))
            k = np.array([(n+1) * a[::t] + j for j in a[::t]]).flatten()
            k=k.astype(int)
            k=np.sort(k)
            ux_k,uy_k = ux[k],uy[k]
            u, p,tri = homogeneousDirichlet(N, 4, f, nu, E)
            conv.append(np.linalg.norm(ux_k - u[::2]))
            h.append(1/(2**i))
           
        order = np.polyfit(np.log(h), np.log(conv), 1)[0]
        self.assertGreater(order, 0.8)
        print("The order of convergence is: ", order)
        
        plt.figure()
        plt.loglog(h,conv)
        plt.xlabel("log(h)")
        plt.ylabel("log(error)")
        plt.savefig("conv.pdf")
        plt.show()
    
    @unittest.skip
    def test_runtime(self):
        highest = 7
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

#Exact strain-vector
def e(x,y):
    e_xx = 2*x*(y**2 - 1)
    e_yy = 2*y*(x**2 - 1)
    e_xy = e_xx + e_yy
    return np.array([e_xx, e_yy, e_xy])
    
class TestStressRecovery(unittest.TestCase):
    
    def test_stress_recovery(self):
        N = 100
        #Find numerical solution
        u, p, tri=homogeneousDirichlet(N,4,f,nu,E)
        
        S, p = StressRecovery(u,p,tri,nu, E)
        
        #Comparing to analytical solution (sigma_xx)
        e_ex = e(p[:,0], p[:,1])
        C =np.array(([1,nu,0],[nu,1,0],[0,0,(1-nu)/2]))*E/(1-nu**2)
        S_ex = C@e_ex
        max_error = np.max(np.abs(S_ex[0]-S[0]))
        self.assertAlmostEqual(max_error, 0, delta=10/N)
        print("Max error of recovered stress is:", max_error, " for N=",N)
        
        #Plotting results
        plot(p[:, 0], p[:, 1], S[0], "sigma_xx, N = "+str(N))
        plot(p[:, 0], p[:, 1], S[1], "sigma_yy, N = "+str(N))
        plot(p[:, 0], p[:, 1], S[2], "sigma_xy, N = "+str(N))
        
        plot(p[:, 0], p[:, 1], S[0]- S_ex[0], "sigma_xx, N = "+str(N))
        plot(p[:, 0], p[:, 1], S[1]- S_ex[1], "sigma_yy, N = "+str(N))
        plot(p[:, 0], p[:, 1], S[2]- S_ex[2], "sigma_xy, N = "+str(N))
        
                

if __name__ == '__main__':
    unittest.main()
