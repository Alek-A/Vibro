# -*- coding: utf-8 -*-
"""
Simulation tool for vibrational analysis
----------------------------------------

Frequency sweeps

Author:
Aleksander Andersen
MSc student in Engineering Design and Applied Mechanics, DTU
aleader@dtu.dk
Created: 28-05-2018

Description:
1) User defines function for ODE and Jacobian matrix (or set jac=None)
2) Transients and post-transients are calculated sequentially
3) Sweeps up and down
4) Uses vibroFun.integrateUser() for which sets up time-parameters
    in terms of Omega-cycles
"""
#%% Initialization
import numpy as np
import time
import matplotlib.pyplot as plt

# Import own functions
import vibroFun


#%% Define ODE
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
beta = 0.05
omega = 1.0
gamma = 1.0
eta = 0.05
p = 0.1
#Omega = 0.7
#par = [eta,gamma,p,omega,beta,Omega]
#parRef = ['eta','gamma','p','omega','beta','Omega']

### Solving parameters
#nSys    = 3          # Dimension of ODE
nDiv    = 100         # Cycles of 2.pi/Omega for determining time step spacing
nCycles = 80         # Cycles to simulate
nSkipTransient = 120 # Cycles of transient to skip

# Set detuning parameter, which controls which Omega to sweep through
sigma = np.linspace(-0.7,0.4)

testOmega = omega*(1+sigma)
nTest = np.size(testOmega)

#%% Sweep up

ampUp = np.array([])

y0 = np.array([0, 0, testOmega[0]]) # Initial conditions

for i in range(nTest):
    Omega = testOmega[i]
    
    ### Simulate transient response
    print('Upsweep, case {0} / {1}'.format(i+1,nTest))
    par = [eta,gamma,p,omega,beta,Omega]
    before = time.time()
    transient = vibroFun.integrateUser(odeFunc,y0,par,Omega,
                                       nDiv=nDiv,nCycles=nSkipTransient, jac=jac)
    after = time.time()
    print('Integrating transient took {:.2f} seconds\n'.format(after-before))
    y0 = transient[1:,-1] # Set new initial value
    
    # Simulate post-transient response
    before = time.time()
    ans = vibroFun.integrateUser(odeFunc,y0,par,Omega,
                                 nDiv=nDiv,nCycles=nCycles)
    after = time.time()
    print('Integrating post-transient took {:.2f} seconds\n'.format(after-before))
    y0 = ans[1:,-1] # Set new initial value
    
    # Save amplitude
    ampUp = np.append(ampUp, np.max(np.abs(ans[1,:])))
    
    

#%% Sweep down
ampDown = np.array([])

y0 = np.array([0, 0, testOmega[-1]]) # Initial conditions

for i in range(20):
    Omega = testOmega[::-1][i] # [::-1] reverses the array
    
    print('Down-sweep, case {0} / {1}'.format(i+1,nTest))
    par = [eta,gamma,p,omega,beta,Omega]
    before = time.time()
    transient = vibroFun.integrateUser(odeFunc,y0,par,Omega,
                                       nDiv=nDiv,nCycles=nSkipTransient, jac=jac)
    after = time.time()
    print('Integrating transient took {:.2f} seconds\n'.format(after-before))
    
    y0 = transient[1:,-1] # end of transient
    before = time.time()
    ans = vibroFun.integrateUser(odeFunc,y0,par,Omega,
                                 nDiv=nDiv,nCycles=nCycles)
    after = time.time()
    print('Integrating post-transient took {:.2f} seconds\n'.format(after-before))
    y0 = ans[1:,-1]
    
    # Save amplitude
    ampDown = np.append(ampDown, np.max(np.abs(ans[1,:])))

# Reverse order
ampDown = ampDown[::-1]

#%% Save data
#np.savetxt('freqSweep/ampUp.txt',ampUp)
#np.savetxt('freqSweep/ampDown.txt',ampDown)
#np.savetxt('freqSweep/freq.txt',testOmega)

#%% Plot

### Load test data
#testOmega = np.loadtxt('freqSweep/freq.txt')
#ampUp     = np.loadtxt('freqSweep/ampUp.txt')
#ampDown   = np.loadtxt('freqSweep/ampDown.txt')

fontsize=8
fig,ax = plt.subplots(figsize=(5,3))
ax.set_xlim([0.6, 1.4])
ax.set_xlabel(r'Normalized frequency $\Omega/\omega$',fontsize=fontsize)
ax.set_ylabel(r'Amplitude |y|',fontsize=fontsize)

ax.plot(testOmega/omega, ampUp, 'ok',ms=7,fillstyle='none', label='Increasing sweep')
ax.plot(testOmega[::-1]/omega, ampDown,'+r',ms=7, label='Decreasing sweep')

ax.legend(loc='upper left',fontsize=fontsize)
fig.tight_layout()
fig.show()
#fig.savefig('freqSweep.png')