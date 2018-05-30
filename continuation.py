# -*- coding: utf-8 -*-

"""
Simulation tool for vibrational analysis
----------------------------------------

Functions for continuation of fixed points

Author:
Aleksander Andersen
MSc student in Engineering Design and Applied Mechanics, DTU
aleader@dtu.dk
2018-05-20

-- References --
[1] Ali H. Nayfeh, B. Balachandran, Applied Nonlinear Dynamics, Wiley, 1995
    Ch.6 has math details on continuation and homotopy algorithms

Based on Matlab implementation by Jon Juel Thomsen.
"""

### Init
import numpy as np
import scipy.linalg as LA
import matplotlib.pyplot as plt


### Define system
# Parameters
eta = 0.02
gamma = 0.1
p = 0.1
omega = 1
beta = 0.02
par = [eta,gamma,p,omega,beta]
parNames = ['eta', 'gamma','p','omega','beta']

# Solving parameters
nPoints = 12000 # Number of points
ds = 0.001     # Step length
maxIter = 200  # Max number of Newton-Raphson iterations
abs_error_tol=1e-5

direction = -1
x0 = 4.0  # Continuation parameter (Omega)
y0 = 0.03      # Other parameter (a)
#x0 = 1.354  # Continuation parameter (Omega)
#y0 = 0.1415      # Other parameter (a)
#direction = 1
#x0 = 0.001  # Continuation parameter (Omega)
#y0 = 0.1      # Other parameter (a)

def F(Omega,a, par):
    """
    Equation to be solved
    F = 0
    """
    eta,gamma = par[0],par[1]
    p,omega   = par[2],par[3]
    beta = par[4]
    return (eta/8*a**3 + beta*a)**2 +(3/8*gamma/omega*a**3 +a*(omega-Omega))**2 -(0.5*p/omega)**2

def Fy(Omega,a, par):
    """
    Partial derivative
    ∂F/∂a = 0
    """
    eta,gamma = par[0],par[1]
    p,omega   = par[2],par[3] # p can be omitted
    beta = par[4]
    return (eta/4*a**3 +2*beta*a)*(eta*3/8*a**2 +beta) +(3/4*gamma/omega*a**3 +2*a*(omega-Omega))*(9/8*gamma/omega*a**2 +omega-Omega)

def Fx(Omega,a, par):
    """
    Partial derivative
    ∂F/∂Ω = 0
    """
    return -2*a*(3/8*gamma/omega*a**3 +a*(omega-Omega))

def jacMod(Omega,a, par):
    """
    Jacobian matrix valid for the the stationary solution
    ∂F/∂a and ∂F/∂psi columns
    The psi terms have been eliminated with the stationary mod. equations
    """
    eta,gamma = par[0],par[1]
    p,omega   = par[2],par[3] # p can be omitted
    beta = par[4]
    
    J11 = -3/8*eta*a**2 -beta
    J12 = 3/8*gamma/omega*a**3 -a*(Omega-omega)
    J21 = -9/8*gamma/omega*a +(Omega-omega)/a
    J22 = -eta/8*a**2 -beta
    J = np.array([[J11, J12],
                  [J21, J22]])
    return J

### Function for stability check
def fixedPointStability(jac,x,y,par=()):
    """
    Checks stability of one or more fixed points.
    Supply function to calculate Jacobian matrix as well as parameters
    
    Returns an array of 0 and 1's where 0==unstable, 1==stable
    """
    stab = np.empty(0)
    
    if np.size(x) == 1:
        # Given a single value to check
        J = jac(x,y,par)
        ev = LA.eigvals(J)
        if ev[0].real >= 0 or ev[1].real >= 0:
            stab = 0
        else:
            stab = 1
    else:
        # Given multiple values to check
        for i in range(np.size(x)):
            J = jac(x[i],y[i],par)
            ev = LA.eigvals(J)
            if ev[0].real >= 0 or ev[1].real >= 0:
                stab = np.append(stab, 0)
            else:
                stab = np.append(stab, 1)
    return stab

### Continuation of fixed points
def arcConti(F,Fx,Fy,x0,y0, par=(),jac=None, 
             nPoints=500, ds=0.01, maxIter=100, abs_error_tol=1e-4,
             direction=-1, slope_tol=0):
    """
    Arclength continuation
    x is the continuation parameter (Omega)
    y is the other parameter (a)
    
    -- Parameters --
    F       function to be solved, F(x,y) = 0
                F(x,y,args=())
    Fx      function ∂F/∂x = 0
                Fx(x,y,args=())
    Fy      function ∂F/∂y = 0
                Fy(x,y,args=())
    x0      initial value
    y0      initial value
    args    arguments to be parsed to functions F,Fx,Fy
    jac     (Optional) Jacobian matrix to determine stability
    """
    # Solution bins
    xx = np.empty(0)
    yy = np.empty(0)
    stab = np.array([1])  # Saving the stability
    istab = np.array([0]) # Index point
    
    postTurnBranch = False
    xs_sign = direction # Sign of dx/ds, indicates direction of continuation
    
    for i in range(nPoints):
        # Find z = (dy/ds) / (dx/ds)
        Fy0 = Fy(x0,y0,par)
        z0 = -Fx(x0,y0,par)/Fy0
    
        if postTurnBranch == True:
            # Change direction of continuation if past turning or branch point
            xs_sign *= -1
        
        xs0 = xs_sign/np.sqrt(1+np.inner(z0,z0)) # dx/ds
        ys0 = z0*xs0 # dy/ds by [1](6.1.13)
        
        # Prediction step along tangent
        x1 = x0 +xs0*ds
        y1 = y0 +ys0*ds
        
        # Newton-Raphson iteration
        k = 0 # iteration number
        converged = False
        xk = x1
        yk = y1
        while converged == False:
            k += 1
            if k > maxIter:
                print('Maximum number of iterations exceeded')
                break
            # Eq. [1](6.1.27) - solve for z1
            z1 = -F(xk,yk,par)/Fy(xk,yk,par)
            z2 = -Fx(xk,yk,par)/Fy(xk,yk,par)
            
            # Compute g_k with [1](6.1.21)
            gk = np.inner((yk-y0),ys0) +(xk-x0)*xs0 -ds
            
            dx_kplus1 = -(gk +np.inner(z1,ys0))/(xs0 +np.inner(z2,ys0))
            dy_kplus1 = z1 + z2*dx_kplus1
            
            # Compute next iteration value [1](6.1.22-23)
            r = 1 # relaxation parameter
            newF_smaller = False
            it = 0
            while newF_smaller == False:
                it += 1
                if it > 500:
                    print('loop newF_smaller exceeded 500 iterations')
                    break
                x_kplus1 = xk +r*dx_kplus1
                y_kplus1 = yk +r*dy_kplus1
                newF_smaller = np.abs(F(x_kplus1, y_kplus1,par)) < np.abs(F(xk,yk,par))
                r = r/2
            
            # Check convergence
            abs_error = np.abs(F(x_kplus1, y_kplus1, par))
            converged = abs_error < abs_error_tol
        
            # Update iteration point
            xk = x_kplus1
            yk = y_kplus1
        
        # Check for turning point or branch point (where dF/dy will be 0 or change sign)
        postTurnBranch = Fy(x0,y0,par)*Fy(x_kplus1,y_kplus1,par) < 0
        
        # Update and save fixed point
        x0 = xk
        y0 = yk        
        xx = np.append(xx,xk)
        yy = np.append(yy,yk)
        
        # Check stability
        # Save the indices where stability changes
        if jac != None:
            pStab = fixedPointStability(jac,xk,yk,par)
            if pStab != stab[-1]:
                istab = np.append(istab, i)
            stab = np.append(stab, pStab)
    if jac != None:
        if istab[-1] != nPoints-1:
            istab = np.append(istab,nPoints-1)
        return xx,yy,istab
    else:
        return xx,yy

### Run
xx,yy,stab = arcConti(F,Fx,Fy,x0,y0, par, jac=jacMod,
                      nPoints=nPoints, ds=ds, maxIter=maxIter, direction=direction,
                      abs_error_tol=abs_error_tol)

    
#%% Plotting

fig,ax = plt.subplots(figsize=(8,5))

### Go through 
if np.size(stab) == 0:
    ax.plot(xx[0:stab[0]], yy[0:stab[0]],ls='-',color='k')
if np.size(stab) > 0:
    ind = 1 # Stability of initial point
    for i in range(np.size(stab)-1):
        if ind == 1:
            ax.plot(xx[stab[i]:stab[i+1]], yy[stab[i]:stab[i+1]],ls='-',color='k') # Stable
            ind = 0
        else:
            ax.plot(xx[stab[i]:stab[i+1]], yy[stab[i]:stab[i+1]],ls=':',color='k') # Unstable
            ind = 1
    # Add
    ax.plot([],[],ls='-',color='k',label='Stable')
    ax.plot([],[],ls=':',color='k',label='Unstable')
ax.set_xlabel(r'$\Omega$',fontsize=14)
ax.set_ylabel(r'$a$',fontsize=14)
ax.set_xlim([omega-0.5,omega+0.5])
ax.legend(loc=1,fontsize=12)
plt.text(0.02,0.68,'$\eta={0}$\n $\gamma = {1}$\n $p = {2}$\n $\omega = {3}$\n $\\beta = {4}$'.format(eta,gamma,p,omega,beta),
         transform=ax.transAxes,fontsize=12)
fig.tight_layout()

#fig.savefig('fig/frequency_response.png')