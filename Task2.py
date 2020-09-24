# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 13:37:41 2020

@author: johan
"""
from getdisc import GetDisc
from matplotlib import pyplot as plt

def PlotMesh(N):
    p, tri, edge = GetDisc(N)
    
    fig, ax = plt.subplots()
    for el in tri:
        ax.plot(p[el, 0], p[el, 1], "ro-")
        ax.plot(p[el[[2,0]], 0], p[el[[2,0]], 1], "ro-")
    return ax
    #print("p:", p, "\n tri:", tri, "\n edge:", edge)

ax = PlotMesh(30)
