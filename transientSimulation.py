# -*- coding: utf-8 -*-
"""
Simulation tool for vibrational analysis
----------------------------------------

Simulation of transient reponse

Author:
Aleksander Andersen
MSc student in Engineering Design and Applied Mechanics, DTU
aleader@dtu.dk
29-05-2018

Description:
User defines function for ODE and Jacobian matrix (or set jac=None).

Plots:
    Time Series
    Phase Plane
Uses vibroFun.integrate() for setting integration

Made with Python 3.5
"""
#%% Initialization
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec # Plot layout
import time
from cmath import sqrt # handles sqrt(-1)

# User functions
import vibroFun


#%% Define ODE functions
def odeFunc(t, y, par):
    """
    Equation of motion in first order autonomous form.
    
    Order should be (t,y,par)
    Parameters can be saved in other variables, e.g. eta = par[0]
        or used directly, e.g. par[0] for slightly more efficiency.
    
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
    """
    Define Jacobian matrix.
    """
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


#%% Parameters

### System parameters

# Transient 1
#beta = 0.01
#omega = sqrt(-0.5)
#gamma = 0.1
#eta = 0.1
#p = 0
#Omega = 0

# Transient 2
#beta = 0.01
#omega = sqrt(1)
#gamma = 0.1
#eta = 0.1
#p = 0
#Omega = 0

# Transient 3
beta = 0.01
omega = sqrt(-0.5)
gamma = 0.2
eta = 0.1
p = 0
Omega = 0

par = [eta,gamma,p,omega,beta,Omega]
parRef = ['eta','gamma','p','omega','beta','Omega']

### Solving parameters
nSys    = 3          # Dimension of ODE


### TRANSIENT SIMULATION
t0 = 0    # Start time
t1 = 100  # End time
dt = 0.01 # Time step for points

### Initial value
#y0 = np.array([0, 0.1, Omega])    # Transient 1,2
y0 = np.array([-0.3, 0.2, Omega]) # Transient 3


#%% Solving step

# Integrate transient
before = time.time()
ans = vibroFun.integrate(odeFunc,y0,par,t1,
                         dt=dt, t0=t0, jac=jac)
after = time.time()
print('Integrating transient took {:.2f} seconds\n'.format(after-before))


#%% Plotting

fontsize=10

# Figure and grid specification
fig = plt.figure(figsize=(9,3))
gs = gridspec.GridSpec(1, 3)  # Specify grid
ax1 = plt.subplot(gs[:, :-1]) # Time response
ax2 = plt.subplot(gs[:, -1:])  # Phase plane

### Time response
# Displacement
ax1.plot(ans[0,:],ans[1,:],'-k',label=r'$y$')
# Velocity
#ax1.plot(ans[0,:],ans[2,:],'--k',alpha=0.7,label=r'$\dot{y}$')

ax1.set_xlim([t0,t1])
ax1.set_xlabel('t', fontsize=fontsize)
ax1.set_ylabel('y', fontsize=fontsize)
ax1.grid(axis='y',alpha=0.4)


# Set axis limits for Phase plane and Poincaré
ppxmin = np.min(ans[1,:])
ppxmax = np.max(ans[1,:])
ppymin = np.min(ans[2,:])
ppymax = np.max(ans[2,:])
ppxmin -= 0.2*np.abs(ppxmin) # Extend the limits a bit
ppxmax += 0.2*np.abs(ppxmax)
ppymin -= 0.2*np.abs(ppymin)
ppymax += 0.2*np.abs(ppymax)


### Phase plane
ax2.plot(ans[1,:],ans[2,:],'-k')
ax2.plot(ans[1,0],ans[2,0],'ok',ms=4) # Start point

ax2.set_xlabel(r'$y$',fontsize=fontsize)
ax2.set_ylabel(r'$\dot{y}$',fontsize=fontsize)

ax2.set_xlim([ppxmin, ppxmax])
ax2.set_ylim([ppymin, ppymax])
ax2.grid(axis='both',which='major',alpha=0.4)

fig.tight_layout()
#fig.savefig(transient3.png)