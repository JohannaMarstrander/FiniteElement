# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 10:35:29 2020

@author: johan
"""
from getdisc import GetDisc
import numpy as np
from Task1 import quadrature2D

p, tri, edge = GetDisc(5)
func = lambda x,y,c: x*0 + y*0 + c
#print("p:", p, "\n tri:", tri, "\n edge:", edge)

def createA(N, Nq, mine = True):
    p, tri, edge = GetDisc(N)
    A = np.zeros((len(p), len(p)))
    
    for el in tri: #for each element
        #Find coefficients of H_alpha^k
        C = np.hstack((np.array([[1],[1],[1]]), p[el]))
        b1, b2, b3 = np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1])
        coeff = np.array([np.linalg.solve(C, b1), np.linalg.solve(C, b2), np.linalg.solve(C, b3)])
        
        if mine:
            #Create elemental Matrix Ak:
            area = 0.5*abs(np.linalg.norm(np.cross(p[el[0]]-p[el[1]],p[el[2]]-p[el[1]])))
            A_k = area * np.array([np.dot(coeff[0,1:], coeff[:,1:].T), np.dot(coeff[1,1:], coeff[:,1:].T), np.dot(coeff[2,1:], coeff[:,1:].T)])
    
            #Assembling A
            j = 0
            for i in el:
                A[i,el] = A_k[j]
                j += 1
        else:
            #Create elemental matrix Ak using Gaussian quadrature  + assembly
            A_k = np.zeros((len(el), len(el)))
            for i in range(len(el)):
                for j in range(len(el)):
                    func = lambda x,y: x*0 + y*0 + np.dot(coeff[i,1:], coeff[j,1:])
                    A_k[i,j] = quadrature2D(p[el[0]],p[el[1]],p[el[2]],Nq,func) 
                    A[el[i],el[j]] += A_k[i,j]
            #MEN SERIØST DETTE SUGER JO
        print(A_k)
    return A


print("A:",createA(5,4, False))