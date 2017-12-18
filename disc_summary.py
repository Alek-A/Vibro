# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 13:53:04 2017

@author: Aleks

Modal analysis for annular plate with clamped-free conditions.

Works for n = [0, 1, 2] radial lines and m = [0, 1, 2, 3] circular lines.

Functions
M0(), M1(), M2() :: Frequency matrices for n = [0, 1, 2]
n0det(), n1det(), n2det() :: Determinants of the frequency matrices
Ltof(), ftok(), ktof() :: Various conversions
R() :: Radial deflection shape
coeff() :: Coefficients A,B,C,D

"""

import scipy as sp # science
import numpy as np # arrays and more
import matplotlib.pyplot as plt # plotting
from mpl_toolkits.mplot3d import axes3d # 3d plotting
from matplotlib import cm # heat map
import matplotlib as mpl

b = 17e-3 # [m] Inner radius
a = 80e-3 # [m] Outer radius
E = 206e9 # [Pa] Young's modulus
nu = 0.3 # Poisson's ratio

    
def M0(cg):
    """
    Frequency Matrix for n=0
    """
    alpha = (17e-3)/(80e-3) # b/a
    nu = 0.3
    M0 = np.mat([[sp.special.j0(cg * alpha),sp.special.y0(cg * alpha),sp.special.i0(cg * alpha),sp.special.k0(cg * alpha)],[-sp.special.j1(cg * alpha) * cg * alpha,-sp.special.y1(cg * alpha) * cg * alpha,sp.special.i1(cg * alpha) * cg * alpha,-sp.special.k1(cg * alpha) * alpha * cg],[-cg * (nu - 1) * sp.special.j1(cg) - cg ** 2 * sp.special.j0(cg),-cg * (nu - 1) * sp.special.y1(cg) - cg ** 2 * sp.special.y0(cg),cg * (nu - 1) * sp.special.i1(cg) + cg ** 2 * sp.special.i0(cg),-cg * (nu - 1) * sp.special.k1(cg) + cg ** 2 * sp.special.k0(cg)],[-cg ** 3 * sp.special.j1(cg),-cg ** 3 * sp.special.y1(cg),-cg ** 3 * sp.special.i1(cg),cg ** 3 * sp.special.k1(cg)]])
    return M0
def M1(cg):
    alpha = (17e-3)/(80e-3) # b/a
    nu = 0.3
    M1 = np.mat([[sp.special.j1(cg * alpha),sp.special.y1(cg * alpha),sp.special.i1(cg * alpha),sp.special.k1(cg * alpha)],[sp.special.j0(cg * alpha) * cg * alpha - sp.special.j1(cg * alpha),sp.special.y0(cg * alpha) * cg * alpha - sp.special.y1(cg * alpha),sp.special.i0(cg * alpha) * cg * alpha - sp.special.i1(cg * alpha),-sp.special.k0(cg * alpha) * cg * alpha - sp.special.k1(cg * alpha)],[(-cg ** 2 - 2 * nu + 2) * sp.special.j1(cg) + cg * sp.special.j0(cg) * (nu - 1),(-cg ** 2 - 2 * nu + 2) * sp.special.y1(cg) + cg * sp.special.y0(cg) * (nu - 1),(cg ** 2 - 2 * nu + 2) * sp.special.i1(cg) + cg * sp.special.i0(cg) * (nu - 1),(cg ** 2 - 2 * nu + 2) * sp.special.k1(cg) - cg * sp.special.k0(cg) * (nu - 1)],[(-cg ** 2 + 2 * nu - 2) * sp.special.j1(cg) - cg * sp.special.j0(cg) * (-cg ** 2 + nu - 1),(-cg ** 2 + 2 * nu - 2) * sp.special.y1(cg) - cg * sp.special.y0(cg) * (-cg ** 2 + nu - 1),(cg ** 2 + 2 * nu - 2) * sp.special.i1(cg) - cg * sp.special.i0(cg) * (cg ** 2 + nu - 1),(cg ** 2 + 2 * nu - 2) * sp.special.k1(cg) + cg * sp.special.k0(cg) * (cg ** 2 + nu - 1)]])
    return M1
def M2(cg):
    alpha = (17e-3)/(80e-3) # b/a
    nu = 0.3
    M2 = np.mat([[(-sp.special.j0(cg * alpha) * cg * alpha + 2 * sp.special.j1(cg * alpha)) / cg / alpha,(-sp.special.y0(cg * alpha) * cg * alpha + 2 * sp.special.y1(cg * alpha)) / cg / alpha,(sp.special.i0(cg * alpha) * cg * alpha - 2 * sp.special.i1(cg * alpha)) / cg / alpha,(sp.special.k0(cg * alpha) * cg * alpha + 2 * sp.special.k1(cg * alpha)) / cg / alpha],[(sp.special.j1(cg * alpha) * cg ** 2 * alpha ** 2 + 2 * sp.special.j0(cg * alpha) * cg * alpha - 4 * sp.special.j1(cg * alpha)) / cg / alpha,(sp.special.y1(cg * alpha) * cg ** 2 * alpha ** 2 + 2 * sp.special.y0(cg * alpha) * cg * alpha - 4 * sp.special.y1(cg * alpha)) / cg / alpha,(sp.special.i1(cg * alpha) * cg ** 2 * alpha ** 2 - 2 * sp.special.i0(cg * alpha) * cg * alpha + 4 * sp.special.i1(cg * alpha)) / cg / alpha,(-sp.special.k1(cg * alpha) * cg ** 2 * alpha ** 2 - 2 * sp.special.k0(cg * alpha) * cg * alpha - 4 * sp.special.k1(cg * alpha)) / cg / alpha],[(((nu - 3) * cg ** 2 - 12 * nu + 12) * sp.special.j1(cg) + 6 * sp.special.j0(cg) * (cg ** 2 / 6 + nu - 1) * cg) / cg,(((nu - 3) * cg ** 2 - 12 * nu + 12) * sp.special.y1(cg) + 6 * sp.special.y0(cg) * (cg ** 2 / 6 + nu - 1) * cg) / cg,(((nu - 3) * cg ** 2 + 12 * nu - 12) * sp.special.i1(cg) - 6 * sp.special.i0(cg) * (-cg ** 2 / 6 + nu - 1) * cg) / cg,(((-nu + 3) * cg ** 2 - 12 * nu + 12) * sp.special.k1(cg) - 6 * sp.special.k0(cg) * (-cg ** 2 / 6 + nu - 1) * cg) / cg],[((cg ** 4 - 4 * cg ** 2 * nu + 24 * nu - 24) * sp.special.j1(cg) - 12 * sp.special.j0(cg) * (-cg ** 2 / 6 + nu - 1) * cg) / cg,((cg ** 4 - 4 * cg ** 2 * nu + 24 * nu - 24) * sp.special.y1(cg) - 12 * sp.special.y0(cg) * (-cg ** 2 / 6 + nu - 1) * cg) / cg,((-cg ** 4 - 4 * cg ** 2 * nu - 24 * nu + 24) * sp.special.i1(cg) + 12 * sp.special.i0(cg) * (cg ** 2 / 6 + nu - 1) * cg) / cg,((cg ** 4 + 4 * cg ** 2 * nu + 24 * nu - 24) * sp.special.k1(cg) + 12 * sp.special.k0(cg) * (cg ** 2 / 6 + nu - 1) * cg) / cg]])
    return M2
    
def n0det(cg):
    """
    Determinant for n=0
    """
    alpha = (17e-3)/(80e-3) # b/a
    nu = 0.3
    J1 = sp.special.j1(cg)
    Y1 = sp.special.y1(cg)
    I1 = sp.special.i1(cg)
    K1 = sp.special.k1(cg)
    J0 = sp.special.j0(cg)
    Y0 = sp.special.y0(cg)
    I0 = sp.special.i0(cg)
    K0 = sp.special.k0(cg)
    J1h = sp.special.j1(cg*alpha)
    Y1h = sp.special.y1(cg*alpha)
    I1h = sp.special.i1(cg*alpha)
    K1h = sp.special.k1(cg*alpha)
    J0h = sp.special.j0(cg*alpha)
    Y0h = sp.special.y0(cg*alpha)
    I0h = sp.special.i0(cg*alpha)
    K0h = sp.special.k0(cg*alpha)
    det = 2 * cg ** 5 * (((((nu - 1) * Y1 + Y0 * cg / 2) * K1 - K0 * Y1 * cg / 2) * J0h - cg * (J0 * Y1 - J1 * Y0) * K0h / 2 - Y0h * (((nu - 1) * K1 - K0 * cg / 2) * J1 + J0 * K1 * cg / 2)) * I1h + ((((nu - 1) * Y1 + Y0 * cg / 2) * K1 - K0 * Y1 * cg / 2) * I0h + (((nu - 1) * Y1 + Y0 * cg / 2) * I1 + I0 * Y1 * cg / 2) * K0h + Y0h * cg * (I0 * K1 + I1 * K0) / 2) * J1h + (-cg * (J0 * Y1 - J1 * Y0) * I0h / 2 + (((-nu + 1) * Y1 - Y0 * cg / 2) * I1 - I0 * Y1 * cg / 2) * J0h + Y0h * (((nu - 1) * J1 + J0 * cg / 2) * I1 + J1 * I0 * cg / 2)) * K1h - ((((nu - 1) * K1 - K0 * cg / 2) * J1 + J0 * K1 * cg / 2) * I0h + cg * (I0 * K1 + I1 * K0) * J0h / 2 + (((nu - 1) * J1 + J0 * cg / 2) * I1 + J1 * I0 * cg / 2) * K0h) * Y1h) * alpha
    
    return det

def n1det(cg):
    alpha = (17e-3)/(80e-3) # b/a ratio
    nu = 0.3
    J1 = sp.special.j1(cg)
    Y1 = sp.special.y1(cg)
    I1 = sp.special.i1(cg)
    K1 = sp.special.k1(cg)
    J0 = sp.special.j0(cg)
    Y0 = sp.special.y0(cg)
    I0 = sp.special.i0(cg)
    K0 = sp.special.k0(cg)
    J1h = sp.special.j1(cg*alpha)
    Y1h = sp.special.y1(cg*alpha)
    I1h = sp.special.i1(cg*alpha)
    K1h = sp.special.k1(cg*alpha)
    J0h = sp.special.j0(cg*alpha)
    Y0h = sp.special.y0(cg*alpha)
    I0h = sp.special.i0(cg*alpha)
    K0h = sp.special.k0(cg*alpha)
    det = 8 * cg ** 3 * alpha * (((((-nu + 1) * Y1 + cg * (-cg ** 2 / 4 + nu - 1) * Y0 / 2) * K1 - cg * K0 * ((cg ** 2 / 4 + nu - 1) * Y1 - cg * Y0 * (nu - 1) / 2) / 2) * J0h - cg ** 3 * (J0 * Y1 - J1 * Y0) * K0h / 8 - Y0h * (((-2 * nu + 2) * K1 - cg * K0 * (cg ** 2 / 4 + nu - 1)) * J1 + cg * ((-cg ** 2 / 4 + nu - 1) * K1 + cg * K0 * (nu - 1) / 2) * J0) / 2) * I1h + ((((nu - 1) * Y1 - cg * (-cg ** 2 / 4 + nu - 1) * Y0 / 2) * K1 + cg * K0 * ((cg ** 2 / 4 + nu - 1) * Y1 - cg * Y0 * (nu - 1) / 2) / 2) * I0h + (((nu - 1) * Y1 - cg * (-cg ** 2 / 4 + nu - 1) * Y0 / 2) * I1 - cg * ((cg ** 2 / 4 + nu - 1) * Y1 - cg * Y0 * (nu - 1) / 2) * I0 / 2) * K0h + Y0h * cg ** 3 * (I0 * K1 + I1 * K0) / 8) * J1h + (-cg ** 3 * (J0 * Y1 - J1 * Y0) * I0h / 8 + (((nu - 1) * Y1 - cg * (-cg ** 2 / 4 + nu - 1) * Y0 / 2) * I1 - cg * ((cg ** 2 / 4 + nu - 1) * Y1 - cg * Y0 * (nu - 1) / 2) * I0 / 2) * J0h - Y0h * (((4 * nu - 4) * J1 - 2 * cg * (-cg ** 2 / 4 + nu - 1) * J0) * I1 + cg * ((-cg ** 2 / 2 - 2 * nu + 2) * J1 + cg * J0 * (nu - 1)) * I0) / 4) * K1h + ((((-2 * nu + 2) * K1 - cg * K0 * (cg ** 2 / 4 + nu - 1)) * J1 + cg * ((-cg ** 2 / 4 + nu - 1) * K1 + cg * K0 * (nu - 1) / 2) * J0) * I0h - cg ** 3 * (I0 * K1 + I1 * K0) * J0h / 4 - (((4 * nu - 4) * J1 - 2 * cg * (-cg ** 2 / 4 + nu - 1) * J0) * I1 + cg * ((-cg ** 2 / 2 - 2 * nu + 2) * J1 + cg * J0 * (nu - 1)) * I0) * K0h / 2) * Y1h / 2)
    
    return det

def n2det(cg):
    alpha = (17e-3)/(80e-3) # b/a ratio
    nu = 0.3
    J1 = sp.special.j1(cg)
    Y1 = sp.special.y1(cg)
    I1 = sp.special.i1(cg)
    K1 = sp.special.k1(cg)
    J0 = sp.special.j0(cg)
    Y0 = sp.special.y0(cg)
    I0 = sp.special.i0(cg)
    K0 = sp.special.k0(cg)
    J1h = sp.special.j1(cg*alpha)
    Y1h = sp.special.y1(cg*alpha)
    I1h = sp.special.i1(cg*alpha)
    K1h = sp.special.k1(cg*alpha)
    J0h = sp.special.j0(cg*alpha)
    Y0h = sp.special.y0(cg*alpha)
    I0h = sp.special.i0(cg*alpha)
    K0h = sp.special.k0(cg*alpha)
    det = (((((-8 * nu + 24) * cg ** 4 + 192 * nu ** 2 + 384 * nu - 576) * Y1 - 48 * cg * (cg ** 4 / 12 + cg ** 2 * (nu - 1) + nu ** 2 + 6 * nu - 7) * Y0) * K1 + 48 * cg * K0 * ((cg ** 4 / 12 + (-nu + 1) * cg ** 2 + nu ** 2 + 6 * nu - 7) * Y1 - 4 * cg * Y0 * (nu - 1))) * J1h + ((((8 * nu - 24) * cg ** 4 - 192 * nu ** 2 - 384 * nu + 576) * K1 - 48 * cg * K0 * (cg ** 4 / 12 + (-nu + 1) * cg ** 2 + nu ** 2 + 6 * nu - 7)) * J1 + 48 * cg * ((cg ** 4 / 12 + cg ** 2 * (nu - 1) + nu ** 2 + 6 * nu - 7) * K1 + 4 * cg * K0 * (nu - 1)) * J0) * Y1h - 48 * cg * (((((-nu / 24 + 0.1e1 / 0.8e1) * cg ** 4 + nu ** 2 + 2 * nu - 3) * Y1 - cg * (cg ** 4 / 12 + cg ** 2 * (nu - 1) + nu ** 2 + 6 * nu - 7) * Y0 / 4) * K1 + cg * K0 * ((cg ** 4 / 12 + (-nu + 1) * cg ** 2 + nu ** 2 + 6 * nu - 7) * Y1 - 4 * cg * Y0 * (nu - 1)) / 4) * J0h - cg * (-cg ** 4 / 12 + (nu - 1) ** 2) * (J0 * Y1 - J1 * Y0) * K0h / 4 + Y0h * ((((nu / 6 - 0.1e1 / 0.2e1) * cg ** 4 - 4 * nu ** 2 - 8 * nu + 12) * K1 - cg * K0 * (cg ** 4 / 12 + (-nu + 1) * cg ** 2 + nu ** 2 + 6 * nu - 7)) * J1 + cg * ((cg ** 4 / 12 + cg ** 2 * (nu - 1) + nu ** 2 + 6 * nu - 7) * K1 + 4 * cg * K0 * (nu - 1)) * J0) / 4) * alpha) * I1h + (((((8 * nu - 24) * cg ** 4 - 192 * nu ** 2 - 384 * nu + 576) * Y1 + 48 * cg * (cg ** 4 / 12 + cg ** 2 * (nu - 1) + nu ** 2 + 6 * nu - 7) * Y0) * I1 + 48 * cg * I0 * ((cg ** 4 / 12 + (-nu + 1) * cg ** 2 + nu ** 2 + 6 * nu - 7) * Y1 - 4 * cg * Y0 * (nu - 1))) * K1h - 48 * cg * alpha * (((((-nu / 24 + 0.1e1 / 0.8e1) * cg ** 4 + nu ** 2 + 2 * nu - 3) * Y1 - cg * (cg ** 4 / 12 + cg ** 2 * (nu - 1) + nu ** 2 + 6 * nu - 7) * Y0 / 4) * K1 + cg * K0 * ((cg ** 4 / 12 + (-nu + 1) * cg ** 2 + nu ** 2 + 6 * nu - 7) * Y1 - 4 * cg * Y0 * (nu - 1)) / 4) * I0h + ((((-nu / 24 + 0.1e1 / 0.8e1) * cg ** 4 + nu ** 2 + 2 * nu - 3) * Y1 - cg * (cg ** 4 / 12 + cg ** 2 * (nu - 1) + nu ** 2 + 6 * nu - 7) * Y0 / 4) * I1 - cg * I0 * ((cg ** 4 / 12 + (-nu + 1) * cg ** 2 + nu ** 2 + 6 * nu - 7) * Y1 - 4 * cg * Y0 * (nu - 1)) / 4) * K0h + cg * (-cg ** 4 / 12 + (nu - 1) ** 2) * Y0h * (I0 * K1 + I1 * K0) / 4)) * J1h + (((((-8 * nu + 24) * cg ** 4 + 192 * nu ** 2 + 384 * nu - 576) * J1 - 48 * cg * (cg ** 4 / 12 + cg ** 2 * (nu - 1) + nu ** 2 + 6 * nu - 7) * J0) * I1 - 48 * cg * I0 * ((cg ** 4 / 12 + (-nu + 1) * cg ** 2 + nu ** 2 + 6 * nu - 7) * J1 - 4 * cg * J0 * (nu - 1))) * Y1h + 12 * cg * alpha * (cg * (-cg ** 4 / 12 + (nu - 1) ** 2) * (J0 * Y1 - J1 * Y0) * I0h + ((((-nu / 6 + 0.1e1 / 0.2e1) * cg ** 4 + 4 * nu ** 2 + 8 * nu - 12) * Y1 - cg * (cg ** 4 / 12 + cg ** 2 * (nu - 1) + nu ** 2 + 6 * nu - 7) * Y0) * I1 - cg * I0 * ((cg ** 4 / 12 + (-nu + 1) * cg ** 2 + nu ** 2 + 6 * nu - 7) * Y1 - 4 * cg * Y0 * (nu - 1))) * J0h + ((((nu / 6 - 0.1e1 / 0.2e1) * cg ** 4 - 4 * nu ** 2 - 8 * nu + 12) * J1 + cg * (cg ** 4 / 12 + cg ** 2 * (nu - 1) + nu ** 2 + 6 * nu - 7) * J0) * I1 + cg * I0 * ((cg ** 4 / 12 + (-nu + 1) * cg ** 2 + nu ** 2 + 6 * nu - 7) * J1 - 4 * cg * J0 * (nu - 1))) * Y0h)) * K1h - 12 * cg * (((((nu / 6 - 0.1e1 / 0.2e1) * cg ** 4 - 4 * nu ** 2 - 8 * nu + 12) * K1 - cg * K0 * (cg ** 4 / 12 + (-nu + 1) * cg ** 2 + nu ** 2 + 6 * nu - 7)) * J1 + cg * ((cg ** 4 / 12 + cg ** 2 * (nu - 1) + nu ** 2 + 6 * nu - 7) * K1 + 4 * cg * K0 * (nu - 1)) * J0) * I0h - cg * (-cg ** 4 / 12 + (nu - 1) ** 2) * (I0 * K1 + I1 * K0) * J0h + ((((nu / 6 - 0.1e1 / 0.2e1) * cg ** 4 - 4 * nu ** 2 - 8 * nu + 12) * J1 + cg * (cg ** 4 / 12 + cg ** 2 * (nu - 1) + nu ** 2 + 6 * nu - 7) * J0) * I1 + cg * I0 * ((cg ** 4 / 12 + (-nu + 1) * cg ** 2 + nu ** 2 + 6 * nu - 7) * J1 - 4 * cg * J0 * (nu - 1))) * K0h) * alpha * Y1h

    return det

def Ltof(L):
    """
    Converts L to frequency [Hz]
    L = k/a where k is Helmholtz number
    """
    b = 17e-3 # inner radius [m]
    a = 80e-3 # outer radius [m]
    h = 0.69e-3 # plate thickness [m]
    E = 206e9 # Young's mod [Pa]
    nu = 0.3 # Poisson's ratio 
    rho = 7800 # density [kg/m3]
    D = E/(1-nu**2)*h**3/12 # flexural stiffness B'
    return L**2/a**2 *np.sqrt(D/rho/h)/2/np.pi
def ftok(f):
    """
    Frequency [Hz] to Helmholtz number
    """
    h = 0.69e-3
    E = 206e9
    nu = 0.3
    rho = 7800
    D = E/(1-nu**2)*h**3/12
    w = 2*np.pi*f
    return (w**2*rho*h/D)**(1/4)
def ktof(k):
    """
    Helmholtz number to frequency [Hz]
    """
    h = 0.69e-3
    E = 206e9
    nu = 0.3
    rho = 7800
    D = E/(1-nu**2)*h**3/12
    return np.sqrt(k**4*D/rho/h)/2/np.pi

def R(r,coeff,k,n,m):
    """
    Calculates the radial deflection shape
    r: radial values, e.g. np.linspace(b,a,100)
    coeff: A,B,C,D in an array
    k : k_m , lambda=
    n,m 
    """
    b = 17e-3
    a = 80e-3
    E = 206e9
    nu = 0.3
    #r = np.linspace(b,a,100)
    R = coeff[0]*sp.special.jn(n,k*r) +coeff[1]*sp.special.yn(n,k*r) +coeff[2]*sp.special.iv(n,k*r) +coeff[3]*sp.special.kn(n,k*r)
    return R

def coeff(M,D=1):
    """
    Calculates coefficients A,B,C,D
    """
    alpha = (M[1,0]*M[0,2] -M[0,0]*M[1,2])/(M[0,0]*M[1,1] -M[1,0]*M[0,1])
    beta = (M[1,0]*M[0,3] -M[0,0]*M[1,3])/(M[0,0]*M[1,1] -M[1,0]*M[0,1])
    gamma = M[0,2]/M[0,0] +alpha*M[0,1]/M[0,0]
    eps = M[0,3]/M[0,0] +beta*M[0,1]/M[0,0]
    sigC = M[2,2] -M[2,0]*gamma +M[2,1]*alpha
    sigD = M[2,3] -M[2,0]*eps +M[2,1]*beta
    C = -sigD/sigC *D
    A = sigD/sigC*gamma*D -eps*D
    B = -sigD/sigC*alpha*D +beta*D
    return np.array([A,B,C,D])


# Root estimates from plots of det(M) vs L
est = np.array([[2.309, 5.777, 9.9, 13.9],
                [2.236, 5.965, 9.99, 14.0],
                [2.568, 6.545, 10.44, 15.0]])

# Solve for roots at estimates with fsolve()
act = np.empty(est.shape)
for i in range(est.shape[1]):
    act[0,i] = sp.optimize.fsolve(n0det, est[0,i])
    act[1,i] = sp.optimize.fsolve(n1det, est[1,i])
    act[2,i] = sp.optimize.fsolve(n2det, est[2,i])


#%%
n1 = 0 # n radial lines
n2 = 0 # m circumferential

if n1 == 0:
    M = M0(act[n1,n2])
    print('Determinant: ',n0det(act[n1,n2]))
elif n1 == 1:
    M = M1(act[n1,n2])
    print('Determinant: ',n1det(act[n1,n2]))
elif n1 == 2:
    M = M2(act[n1,n2])
    print('Determinant: ',n2det(act[n1,n2]))

xdiv = 50
ydiv = 50
scaling = 5
r = np.linspace(b+1e-3,a,xdiv)
theta = np.linspace(0,2*np.pi,ydiv)
freq = Ltof(act[n1,n2])
X,Y = np.meshgrid(r,theta)
X1,Y1 = X*np.cos(Y), X*np.sin(Y) # rectangular mesh

Z = R(X,coeff(M,D=scaling),act[n1,n2]/a,n1,n2) *np.cos(n1*Y)

### 3D plot
fig = plt.figure(figsize=(16,10))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X1,Y1,Z,
                 rstride=1, cstride=2,cmap=cm.YlGnBu_r,edgecolors='grey')
ax.plot_wireframe(X1,Y1,Z,rstride=1, cstride=2,color='k',alpha=0.1)

ax.set_zlim([-10,10])
ax.set_title(r'$(m,n)=({0:.0f},{1:.0f})$ - $f={2:.1f}$ Hz'.format(n2,n1,freq))

### Contour plot
fig,ax = plt.subplots()
C1 = plt.Circle((0,0), a, color='teal',alpha=0.25)
C2 = plt.Circle((0,0), b, color='white')
ax.add_patch(C1)
ax.add_patch(C2)
CS = plt.contour(X1,Y1,Z,[0],colors='black',ls='--')
plt.axis('equal')
ax.set_title(r'$(m,n)=({0:.0f},{1:.0f})$ - $f={2:.1f}$ Hz'.format(n2,n1,freq))