# -*- coding: utf-8 -*-
"""
Simulation tool for vibrational analysis
----------------------------------------

Module containing functions used for utility
(plotting, loading bar etc.)

Aleksander Andersen
MSc student in Engineering Design and Applied Mechanics, DTU
aleader@dtu.dk
26-05-2018
"""
#%% Imports
import sys
import numpy as np
#%% Functions

# update_progress() : Displays or updates a console progress bar
## Accepts a float between 0 and 1. Any int will be converted to a float.
## A value under 0 represents a 'halt'.
## A value at 1 or bigger represents 100%
def update_progress(progress):
    """
    # update_progress() : Displays or updates a console progress bar
    Accepts a float between 0 and 1. Any int will be converted to a float.
    A value under 0 represents a 'halt'.
    A value at 1 or bigger represents 100%
    """
    barLength = 20 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rPercent: [{0}] {1}% {2}".format( "="*block + " "*(barLength-block), progress*100, status)
    sys.stdout.write(text)
    sys.stdout.flush()
    

### Trigonometric helper functions
def _F(u):
    # Used for beamModes()
    return np.sinh(u) +np.sin(u)
def _G(u):
    # Used for beamModes()
    return np.cosh(u) +np.cos(u)
def _H(u):
    # Used for beamModes()
    return np.sinh(u) -np.sin(u)
def _J(u):
    # Used for beamModes()
    return np.cosh(u) -np.cos(u)

def beamModes(x,j,case):
    """
    Calculates mode shape j for a flexurally vibrating uniform beam.
    
    -- Reference --
    Table D.3, Vibrations and Stability by J J Thomsen
    
    -- Parameters --
    x :: np.array 
         Dimensionless position on beam (0 to 1)
    j :: int > 0
         Mode number
    case :: int
            0 : Clamped-clamped
            1 : Clamped-hinged
            2 : Clamped-free
            3 : Clamped-guided
    
    -- Returns --
    shape :: np.array for the mode shape
    """
    # Check for errors in parameters
    if np.min(x)<0 or np.max(x)>1:
        print('x is not normalized.')
    if type(j) != int or j < 1:
        print('j-parameter is wrong.')
        raise ValueError
    if type(case) != int or case<0 or case>3:
        print('case-parameter is wrong.')
        raise ValueError
    
    # Clamped-clamped
    if case == 0:
        if 0 < j < 5:
            ev = np.array([4.73,7.8532,10.9956,14.1372])[j-1]
        elif j > 4:
            ev = (2*j+1)*np.pi/2
        shape = _J(x*ev) - _H(x*ev)*_J(ev)/_H(ev)
    # Clamped-hinged
    elif case == 1:
        if 0 < j < 5:
            ev = np.array([3.9266,7.0686,10.2102,13.3518])[j-1]
        elif j > 4:
            ev = (4*j+1)*np.pi/4
        shape = _J(x*ev) - _H(x*ev)*_J(ev)/_H(ev) # same as case 0
    # Clamped-free
    elif case == 2:
        if 0 < j < 5:
            ev = np.array([1.8751,4.6941,7.8548,10.9955])[j-1]
        elif j > 4:
            ev = (2*j-1)*np.pi/2
        shape = _J(x*ev) - _H(x*ev)*_G(ev)/_F(ev)
    # Clamped-guided
    elif case == 3:
        if 0 < j < 5:
            ev = np.array([2.3650,5.4978,8.6394,11.7810])[j-1]
        elif j > 4:
            ev = (4*j-1)*np.pi/4
        shape = _J(x*ev) - _H(x*ev)*_F(ev)/_J(ev)
    return shape

def _pendulumCoord(y_disp, xg,yg):
    """
    Used for a plotting implementation.
    Takes the displacement (y_disp) of the beam tip
    along with a mode shape (xg,yg)
    
    
    Example
    > x = np.linspace(0,1,10)
    > 
    >
    
    1) Normalize
    2) Convert into x,y-coordinates
    y --> ±x
    x --> (0,-1)y
    
    -- Parameters --
    y_disp :: displacement of tip point
    xg :: array, 0 to 1 for corresponding yg
    yg :: array, mode shape corresponding to xg
    """
    y2 = -xg # goes from 0 to -1
    x2 = y_disp/yg[-1] * yg
    iix = np.array([np.arange(0,len(yg)-1), np.arange(1,len(yg))]).T
#    return x2,y2
    return np.stack((x2,y2)).T, iix