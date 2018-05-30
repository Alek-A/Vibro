# -*- coding: utf-8 -*-
"""
Simulation tool for vibrational analysis
----------------------------------------

Module containing functions for the analysis

Aleksander Andersen
MSc student in Engineering Design and Applied Mechanics, DTU
aleader@dtu.dk
26-05-2018
"""
#%% Imports
import numpy as np
from scipy.integrate import ode
from utilityFun import update_progress

def readFileName(fn,val='Omega'):
    if val == 'Omega':
        Omega = float(fn[5:9].replace('-','.'))
    return Omega

def filterTransient(y,index=None,time=None):
    t = y[0,:]
    if index != None and time != None:
        print('Only use index or time. Not both')
    elif type(index) == int:
        # Slice by index
            return y[:,index:]
    elif time != None:
        # Slice by time
            return y[:,t>=time]

def integrate(odeFunc,y0,par,t1,
              dt=0.01, t0=0, jac=None):
    """
    
    
    odeFunc  function for ODE, func(t,y, args=())
    jac      function that returns jacobian matrix
    y0       array, initial conditions
    par      parameters to pass to myODE
    t1       end time
    dt       time step
    t0       start time
    
    returns
    ans     array (n+1, time) of results
    
    Notes on further use:
        scipy.integrate.ode can only have one class instance at a given time,
        as they are not reentrant. This function only returns the results
    """
    r = ode(odeFunc, jac=jac).set_integrator('dopri5',nsteps=500,rtol=1e-8)
    r.set_initial_value(y0, t0)
    r.set_f_params(par)
    r.set_jac_params(par)
    ans = np.concatenate((np.array([t0]),y0))[:,None]
    while (r.t < t1) and r.successful():
        update_progress(r.t/t1)
#        print('Ω: {0:.2f} -> {1:.2f} %'.format(Omega,r.t/t1*100))
        r.integrate(r.t+dt)
        ans = np.append(ans, np.concatenate((np.array([r.t]), r.y))[:,None],axis=1)
        
    print('')
    return ans

def integrateUser(odeFunc,y0,par,Omega,
                  nDiv=50, nCycles=40, t0=0, jac=None):
    """
    A wrapper for using integrate.
    Calculates necessary dt, t1 in terms of Omega-cycles.
    """
    dt = 2*np.pi/Omega/nDiv
    t1 = nCycles*2*np.pi/Omega + t0
    return integrate(odeFunc,y0,par,t1,dt=dt,t0=t0, jac=jac)


def poincareMap(t,x,y,Omega,multiple=None,startIndex=0,returnTime=False):
    """
    Computes values for a Poincaré map by slicing the array of state variables.
    
    If given 'multiple', this value will be used to slice the arrays.
    If not given 'multiple', the integer value will be estimated.
    The function will not run if 'multiple' cannot be estimated.
    
    The corresponding time values can be returned by setting returnTime=True
    
    Parameters
    -----------
    t           array, time series
    x           array, state variable for x-axis
    y           array, state variable for y-axis
    Omega       float, Excitation frequency, [rad/s] 
                    or equivalent to time series
    multiple    is time steps and excitation period integer related?
                    None (default): will check. Otherwise interpolate
                    0: will not check
                    int: supplied integer spacing will be used
    startIndex  index value to start at, defaults to 0
    """
    t2 = t-t[0]          # Time starting at 0
    Tp = 2*np.pi/Omega  # Poincare sampling time
    dt = t2[1]      # Time step
    # Check if time steps is a integer multiple of Tp
    if multiple == None:
        if  int(np.round(Tp/dt,2)) == np.round(Tp/dt,2):
            multiple = int(np.round(Tp/dt,2))
            print('Found multiple: ',multiple)
    
    if multiple != 0 and multiple != None:
        X,Y = x[startIndex::multiple], y[startIndex::multiple]
        T = t[startIndex::multiple]
    
    if multiple == None or multiple == 0:
        print('Interpolation scheme not yet invented.\n Please wait for the future')
#        from scipy.interpolate import interp1d
#        N = np.size(t) # number of points
#        runYetAgain = True
#        i = 1
#        Tpc = Tp
#        while runYetAgain == True:
#            if t[i] > Tpc:
#                overshoot = (t[i]-Tpc)/dt
#                interp
    if returnTime == True:
        return X,Y,T
    else:
        return X,Y

def lyapLocal(t, eps):
    num   = np.linalg.norm(eps[:, 1:], axis=0) # |eps(t_k+1)|
    denum = np.linalg.norm(eps[:,:-1], axis=0) # |eps(t_k)|
    return np.log2(num/denum)/(t[1:] -t[:-1])

def lyapGlobal(localLyap, return_all=True):
    if return_all:
        globalLyap = np.empty(np.size(localLyap))
        for i in range(np.size(localLyap)):
            globalLyap[i] = np.mean(localLyap[:i+1])
    else:
        globalLyap = np.mean(localLyap)
    return globalLyap
