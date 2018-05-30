# -*- coding: utf-8 -*-
"""
Simulation tool for vibrational analysis
----------------------------------------

Simulation of periodic response

Author:
Aleksander Andersen
MSc student in Engineering Design and Applied Mechanics, DTU
aleader@dtu.dk
28-05-2018

Description:
User defines functions for ODE, Jacobian matrix (or set jac=None),
    and an extended ODE with variational equations. See eq. [1](6.5)
The integration is sequential in three steps:
    1) Transient
    2) Small transient for extended system (for Lyap. exponents)
    3) Periodic / post-transient response
Plots:
    Post-transient time response
    Phase plane
    Poincaré Maps
    Largest Lyapunov exponent (after eq. [1](6.5 - 6.7))
    Frequency spectrum
Uses vibroFun.integrateUser() which sets up time-parameters
    in terms of Omega-cycles
    
References
[1] Vibrations and Stability, Jon Juel Thomsen, 2.ed., Springer, 2003
    Estimation of Lyapunov exponent, Poincaré Maps, and more.
"""
#%% Initialization
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec # Plot layout
from cmath import sqrt   # handles sqrt(-1)
from scipy import signal # FFT

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

def extendedODE(t,y, par):
    """
    Like odeFunc() but extended with variational equations.
    Used for estimating Lyapunov exponents.
    
    y[0:2] are the state space
    y[3:5] are the variational states
    """
    eta   = par[0]
    gamma = par[1]
    p     = par[2]
    omega = par[3]
    beta  = par[4]
    Omega = par[5]
    # Linearize system around current state/trajectory
    J = np.array([[0, 1, 0],
                  [-omega**2 -3*gamma*y[0]**2 -2*eta*y[0]*y[1], -2*beta -eta*y[0]**2, p*np.cos(y[2])],
                  [0, 0, 0]])
    yd = np.array([y[1],
                   -2*beta*y[1] -omega**2*y[0] -gamma*y[0]**3 -eta*y[0]**2*y[1] +p*np.sin(y[2]),
                   Omega,
                   J[0,0]*y[3] + J[0,1]*y[4] + J[0,2]*y[5],
                   J[1,0]*y[3] + J[1,1]*y[4] + J[1,2]*y[5],
                   J[2,0]*y[3] + J[2,1]*y[4] + J[2,2]*y[5]])
    return yd


#%% Parameters

### System parameters
beta = 0.01
omega = sqrt(-0.5)
gamma = 0.4
eta = 0.05
p = 0.2
Omega = 0.82
fn = 'Omega' + str(Omega).replace('.','_')

par = [eta,gamma,p,omega,beta,Omega]
parRef = ['eta','gamma','p','omega','beta','Omega']

### Solving parameters
nSys    = 3          # Dimension of ODE
nDiv    = 100        # Cycles of 2.pi/Omega for determining time step spacing

### PERIODIC SOLUTION
nSkipTransient = 300 # Cycles of transient to skip
nSkipLyap = 40       # Skipping the transient on the variational part
nCycles = 250        # Cycles to simulate

### Initial value
y0 = np.array([0, 0, Omega]) # Initial conditions


#%% Solving step
# Integrate transient
before = time.time()
transient = vibroFun.integrateUser(odeFunc,y0,par,Omega,
                                   nDiv=nDiv,nCycles=nSkipTransient, jac=jac)
after = time.time()
print('Integrating transient took {:.2f} seconds\n'.format(after-before))

# Set new initial conditions for variational system
# Unit-normalized 
y0 = np.append(transient[1:,-1], np.ones(nSys)/np.sqrt(nSys))

### Solve a bit post-transient
before = time.time()
transient2 = vibroFun.integrateUser(extendedODE,y0,par,Omega,
                                    nDiv=nDiv,nCycles=nSkipLyap)
after = time.time()
print('Integrating small extra transient took {:.2f} seconds\n'.format(after-before))

### Solve the rest
y0 = transient2[1:,-1]
before = time.time()
ans = vibroFun.integrateUser(extendedODE,y0,par,Omega,
                                    nDiv=nDiv,nCycles=nCycles)
after = time.time()
print('Integrating post-transient took {:.2f} seconds\n'.format(after-before))

#%% Post processing

### Points for Poincaré Map
X,Y = vibroFun.poincareMap(ans[0,:],ans[1,:],ans[2,:],Omega, multiple=nDiv,
                           startIndex=0,returnTime=False)


localLyap  = vibroFun.lyapLocal(ans[0,:], ans[-3:,:])
globalLyap = vibroFun.lyapGlobal(localLyap)
lyap = globalLyap[-1]

### FFT of periodic time signal
fs = 1/(ans[0,1]-ans[0,0]) # Sampling frequency
nperseg = ans.shape[1]     # Use whole periodic dataset
f,Sxx = signal.csd(ans[1,:],ans[1,:],fs,window='hanning',
                   nperseg=nperseg,noverlap=0,axis=0)

#%% Plotting
fontsize=12

#fig = plt.figure(figsize=(16,10))
fig = plt.figure(figsize=(12,8))
gs = gridspec.GridSpec(3, 3)  # Specify grid
ax1 = plt.subplot(gs[:1, :])  # Time response
ax2 = plt.subplot(gs[1, 0])   # Phase plane
ax3 = plt.subplot(gs[1, 1])   # Poincaré
ax4 = plt.subplot(gs[1, 2])   # Lyapunov exponent
ax5 = plt.subplot(gs[2, :-1]) # FFT

### Time response
ax1.set_title('Time series')
ax1.plot(ans[0,:],ans[1,:],'-')
ax1.set_xlabel('t',fontsize=fontsize)
ax1.set_ylabel('y',fontsize=fontsize)
ax1.grid(alpha=0.2)
ax1.set_xlim(xmin=0)

# Set axis limits for Phase plane and Poincaré
ppxmin = np.min(ans[1,:])
ppxmax = np.max(ans[1,:])
ppymin = np.min(ans[2,:])
ppymax = np.max(ans[2,:])
ppxmin -= 0.1*np.abs(ppxmin) # Extend the limits a bit
ppxmax += 0.1*np.abs(ppxmax)
ppymin -= 0.1*np.abs(ppymin)
ppymax += 0.1*np.abs(ppymax)


### Phase plane
ax2.set_title('Phase plane')
ax2.plot(ans[1,:],ans[2,:],'-',lw=1.2)
ax2.set_xlabel(r'$y$',fontsize=fontsize)
ax2.set_ylabel(r'$\dot{y}$',fontsize=fontsize)
ax2.set_xlim([ppxmin, ppxmax])
ax2.set_ylim([ppymin, ppymax])
ax2.grid(alpha=0.3)

### Poincaré Map
ax3.set_title('Poincaré map')
ax3.plot(X,Y,'o',ms=3)
ax3.set_xlabel(r'$y$',fontsize=fontsize)
ax3.set_ylabel(r'$\dot{y}$',fontsize=fontsize)
ax3.set_xlim([ppxmin, ppxmax])
ax3.set_ylim([ppymin, ppymax])
ax3.grid(alpha=0.3)


### Lyapunov exponent
ax4.set_title('Largest Lyapunov exponent')
ax4.plot(ans[0,1:], globalLyap,label=r'$\hat{\lambda_1}$ = '+'{:.6f}'.format(lyap))
ax4.set_xlabel('t',fontsize=fontsize)
ax4.set_ylabel(r'$\hat{\lambda}^{[N]}$',fontsize=fontsize)
ax4.legend()
ax4.grid(alpha=0.3)

### FFT
ax5.set_title('Frequency spectrum')
ax5.plot(f*2*np.pi,20*np.log10(Sxx))
ax5.set_xlabel(r'$\omega$',fontsize=fontsize)
ax5.set_ylabel('FFT(y) [dB]',fontsize=fontsize)
ax5.grid(alpha=0.3)

fig.tight_layout()
#fig.savefig('fig/'+fn+'.png')
#np.savetxt('sim/'+fn+'.txt', np.vstack((X,Y)))