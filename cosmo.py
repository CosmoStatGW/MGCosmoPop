#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 17:00:13 2021

@author: Michi
"""
import numpy as np
from astropy.cosmology import FlatwCDM, FlatLambdaCDM
from scipy import interpolate
#from astropy import constants as const
#import astropy.units as u
from scipy.optimize import fsolve
from glob import *


cosmoglob = FlatwCDM(H0=H0GLOB, Om0=Om0GLOB, w0=-1)
zGridGLOB = np.concatenate([ np.logspace(start=-15, stop=np.log10(9.99e-09), base=10, num=10), np.logspace(start=-8, stop=np.log10(7.99), base=10, num=1000), np.logspace(start=np.log10(8), stop=5, base=10, num=100)])
dLGridGLOB = cosmoglob.luminosity_distance(zGridGLOB).to(which_unit).value



######################
# FUNCTIONS FOR COSMOLOGY
######################

def uu(z, Om, w0):
    '''
    Dimensionless comoving distance. Does not depend on H0
    '''
    if w0!=-1:
        return 70/clight*FlatwCDM(H0=70, Om0=Om, w0=w0, Neff=0).comoving_distance(z).value
    else:
        return 70/clight*FlatLambdaCDM(H0=70, Om0=Om).comoving_distance(z).value

def E(z, Om, w0):
    '''
    E(z). Does not depend on H0
    '''
    if w0!=-1:
        return FlatwCDM(H0=70, Om0=Om, Neff=0).efunc(z)
    else:
        return FlatLambdaCDM(H0=70, Om0=Om).efunc(z)

#def j(z, Om, w0):
#    '''
#    Jacobian of comoving volume, dimensioneless. Does not depend on H0
#    '''
#    return 4*np.pi*FlatwCDM(H0=70, Om0=Om, w0=w0).differential_comoving_volume(z).value*(70/clight)**3


def dV_dz(z, H0, Om, w0):
    '''
    Jacobian of comoving volume, with correct dimensions [Mpc^3]. Depends on H0
    '''
    if w0!=-1:
        return 4*np.pi*FlatwCDM(H0=H0, Om0=Om, w0=w0, Neff=0).differential_comoving_volume(z).value
    else:
        return 4*np.pi*FlatLambdaCDM(H0=H0, Om0=Om,).differential_comoving_volume(z).value

def s(z, Xi0, n):
    return (1+z)*Xi(z, Xi0, n)

def sPrime(z, Xi0, n):
    return Xi(z, Xi0, n)-n*(1-Xi0)/(1+z)**n



def ddL_dz(z, H0, Om, w0, Xi0, n):
    '''
     Jacobian d(DL)/dz  [Mpc]
    '''
    return (sPrime(z, Xi0, n)*uu(z, Om, w0)+s(z, Xi0, n)/E(z, Om, w0))*(clight/H0)


def dLGW(z, H0, Om, w0, Xi0, n):
    '''                                                                                                          
    Modified GW luminosity distance in units set by utils.which_unit (default Mpc)                                                                           
    '''
    if w0!=-1:
        cosmo=FlatwCDM(H0=H0, Om0=Om, w0=w0, Neff=0)
    else:
        cosmo=FlatLambdaCDM(H0=H0, Om0=Om)
    return (cosmo.luminosity_distance(z).to(which_unit).value)*Xi(z, Xi0, n)


def Xi(z, Xi0, n):
    return Xi0+(1-Xi0)/(1+z)**n


######################
# SOLVERS FOR DISTANCE-REDSHIFT RELATION
######################

def z_from_dLGW_fast(r, H0, Om, w0, Xi0, n):
    '''
    Returns redshift for a given luminosity distance r (in Mpc by default). Vectorized
    '''
    if (Om==Om0GLOB) & (w0==-1.):
        dLGrid = dLGridGLOB/H0*H0GLOB
    else:
        if w0==-1:
            cosmo = FlatLambdaCDM(H0=H0, Om0=Om, Neff=0)
        else:
            cosmo = FlatwCDM(H0=H0, Om0=Om, w0=w0, Neff=0)
        dLGrid = cosmo.luminosity_distance(zGridGLOB).to(which_unit).value
    z2dL = interpolate.interp1d( dLGrid*Xi(zGridGLOB, Xi0, n), zGridGLOB, kind='cubic', bounds_error=False, fill_value=(0,0.), assume_sorted=True)
    return z2dL(r)

def z_from_dLGW(dL_GW_val, H0, Om, w0, Xi0, n):
    '''Returns redshift for a given luminosity distance dL_GW_val (in Mpc by default)                                         '''
    func = lambda z : dLGW(z, H0, Om, w0, Xi0, n) - dL_GW_val
    z = fsolve(func, 0.5)
    return z[0]

