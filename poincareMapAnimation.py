# -*- coding: utf-8 -*-
"""
Simulation tool for vibrational analysis
----------------------------------------

Animation of stretching/folding of the Poincaré map

Aleksander Andersen
MSc student in Engineering Design and Applied Mechanics, DTU
aleader@dtu.dk
26-05-2018
"""
#%% Imports
import numpy as np
from vibroFun import *
import matplotlib.pyplot as plt
import os
import imageio
#%% Define system
def myODE(t, y, par):
    """
    Equation of motion in first order autonomous form
    Course exercise, eq.9
    
    Single mode approximation
    
    y[0] : displacement
    y[1] : velocity
    y[2] : Omega.t
    """
    eta   = par[0]
    gamma = par[1]
    p     = par[2]
    omega = par[3]
    beta  = par[4]
    Omega = par[5]
    
    yd = np.array([y[1],
                   -2*beta*y[1] -omega**2*y[0] -gamma*y[0]**3 -eta*y[0]**2*y[1] +p*np.sin(y[2]),
                   Omega])
    return yd

def jac(t,y, par):
    eta   = par[0]
    gamma = par[1]
    p     = par[2]
    omega = par[3]
    beta  = par[4]
    #Omega = par[5] # not used in jacobian
    """
    Jacobi matrix of myODE()
    """
    J = np.array([[0, 1, 0],
                  [-omega**2 -3*gamma*y[0]**2 -2*eta*y[0]*y[1], -2*beta -eta*y[0]**2, p*np.cos(y[2])],
                  [0, 0, 0]])
    return J

### System parameters
beta = 0.05
omega = -0.707j
gamma = 1.0
eta = 0.05
p = 0.1
Omega = 0.7
par = [eta,gamma,p,omega,beta,Omega]
parRef = ['eta','gamma','p','omega','beta','Omega']


### Solving parameters
nIntMultiple = 100   # time step spacing
nCycles = 500       # cycles to simulate
nSkipTransient = 50 # cycles to skip

y0 = np.array([0, 0, Omega]) # Initial conditions


### Solve / integrate
ans = integrateUser(myODE,jac,y0,par,Omega,
                    nIntMultiple=nIntMultiple,nCycles=nCycles)

#%% Save images

folder = './fig_ani/02_'
title  = r'$\eta={0}, \gamma={1}, p={2}, \omega={3}, \beta={4}, \Omega={5}$'.format(par[0],
                  par[1],par[2],par[3],par[4],par[5])
fontsize = 14
Tp = nSkipTransient*2*np.pi/Omega
y2 = filterTransient(ans, time=Tp)

i = 0
while i < nIntMultiple:
    X,Y = poincareMap(y2[0,:],y2[1,:],y2[2,:],Omega,multiple=nIntMultiple,
                      startIndex=i,returnTime=False)
    plt.subplots(figsize=(7,5))
    plt.title(title)
    plt.plot(X,Y,'.k',ms=2)
    plt.xlabel(r'$y$',fontsize=fontsize)
    plt.ylabel(r'$\dot{y}$',fontsize=fontsize)
    plt.xlim([np.min(y2[1,:]),np.max(y2[1,:])])
    plt.ylim([np.min(y2[2,:]),np.max(y2[2,:])])
    plt.tight_layout()
    plt.savefig(folder+str(i)+'.png')
    plt.close()
    i += 1
    update_progress(i/nIntMultiple)
print('')
#%% Animation
#images = []
#for i in range(nIntMultiple):
#    file = folder+str(i)+'.png'
#    images.append(imageio.imread(file))
#            
#
#imageio.mimsave('./poincare_animation.gif', images[2::], duration=5)