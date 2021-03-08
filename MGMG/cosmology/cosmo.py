#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 11:34:47 2021

@author: Michi
"""

import numpy as np
from astropy.cosmology import FlatwCDM, FlatLambdaCDM, Planck15
from scipy import interpolate
from astropy import constants as const
import astropy.units as u
from scipy.optimize import fsolve




class Cosmo(object):
    
    def __init__(self, dist_unit = u.Gpc, baseValues=None):
        
        self.dist_unit=dist_unit
        self.params = ['H0', 'Om', 'w0', 'Xi0', 'n']
        self.n_params = len(self.params)
        if baseValues is None:
            self.baseValues = {'H0': Planck15.H0.value ,
                           'Om' : Planck15.Om0 , 
                           'w0': -1, 
                           'Xi0': 1, 
                           'n': 1.91}
        else:
            self.baseValues=baseValues
        
        self.names = {    'H0':r'$H_0$', 
                          'Om':r'$\Omega_{\rm {m,}0 }$',
                          'w0':r'$w_{0}$',
                           'Xi0':r'$\Xi_0$', 
                           'n':r'$n$', }
        
        self.cosmoGlobals = FlatwCDM( H0=self.baseValues['H0'], Om0=self.baseValues['Om'], w0=self.baseValues['w0'])
        self.zGridGlobals = np.concatenate([ np.logspace(start=-15, stop=np.log10(9.99e-09), base=10, num=10), np.logspace(start=-8, stop=np.log10(7.99), base=10, num=1000), np.logspace(start=np.log10(8), stop=5, base=10, num=100)])
        self.dLGridGlobals = self.cosmoGlobals.luminosity_distance(self.zGridGlobals).to(dist_unit).value

        self.clight=const.c.value*1e-03 # c in km/s
        
    
    
    def _set_values(self, values_dict):
            print('cosmo basevalues: %s' %str(self.baseValues))
            for key, value in values_dict.items():
                if key in self.baseValues:
                    self.baseValues[key] = value
                    print('Setting value of %s to %s in %s' %(key, value, self.__class__.__name__))
                 
    
    #def set_param(self, paramValues):
    #    for param in paramValues.keys():
    #        self.baseValues[param] = paramValues[param]
    #        print('Set %s to %s' %(param, paramValues[param]))
        
    
    def _get_all_values(self, Lambda):
        H0, Om, w0, Xi0, n = Lambda
        return H0, Om, w0, Xi0, n
    
    def _get_values(self, Lambda, names):
        H0, Om, w0, Xi0, n =self._get_all_values(Lambda)
        vals=[]
        for i in range( self.n_params):
            for j,name in enumerate(names):
                if self.params[i]==name:
                    vals.append(Lambda[i])
        return vals
    
    
    

    ######################
    # FUNCTIONS FOR COSMOLOGY
    ######################

    def uu(self, z, Om, w0):
        '''
        Dimensionless comoving distance. Does not depend on H0
        '''
        if w0!=-1:
            return 70/self.clight*FlatwCDM(H0=70, Om0=Om, w0=w0, Neff=0).comoving_distance(z).to(u.Mpc).value
        else:
            return 70/self.clight*FlatLambdaCDM(H0=70, Om0=Om).comoving_distance(z).to(u.Mpc).value

    def E(self,z, Om, w0):
        '''
        E(z). Does not depend on H0
        '''
        if w0!=-1:
            return FlatwCDM(H0=70, Om0=Om, w0=w0, Neff=0).efunc(z)
        else:
            return FlatLambdaCDM(H0=70, Om0=Om).efunc(z)



    def dV_dz(self,z, H0, Om, w0):
        '''
        Jacobian of comoving volume, with correct dimensions [Mpc^3]. Depends on H0
        '''
        if w0!=-1:
            return 4*np.pi*FlatwCDM(H0=H0, Om0=Om, w0=w0, Neff=0).differential_comoving_volume(z).to(self.dist_unit**3/u.sr).value
        else:
            return 4*np.pi*FlatLambdaCDM(H0=H0, Om0=Om,).differential_comoving_volume(z).to(self.dist_unit**3/u.sr).value

    def log_dV_dz(self, z, H0, Om0, w0):
        res =  np.log(4*np.pi)+3*np.log(self.clight)-3*np.log(H0)+2*np.log(self.uu(z, Om0, w0))-np.log(self.E(z, Om0, w0))
        if self.dist_unit==u.Gpc:
            res -=9*np.log(10)
        return res


    def s(self, z, Xi0, n):
        return (1+z)*self.Xi(z, Xi0, n)
    
    def sPrime(self, z, Xi0, n):
        return self.Xi(z, Xi0, n)-n*(1-Xi0)/(1+z)**n



    def ddL_dz(self, z, H0, Om, w0, Xi0, n):
        '''
        Jacobian d(DL)/dz  [Mpc]
        '''
        if self.dist_unit==u.Gpc:
            H0*=1e03
        return (self.sPrime(z, Xi0, n)*self.uu(z, Om, w0)+self.s(z, Xi0, n)/self.E(z, Om, w0))*(self.clight/H0)


    def log_ddL_dz(self, z, H0, Om0, w0, Xi0, n):
        res =  np.log(self.clight)-np.log(H0)+np.log(self.sPrime(z, Xi0, n)*self.uu(z, Om0, w0)+self.s(z, Xi0, n)/self.E(z, Om0, w0))
        if self.dist_unit==u.Gpc:
            res -= 3*np.log(10)
        return res

    def dLGW(self, z, H0, Om, w0, Xi0, n):
        '''                                                                                                          
        Modified GW luminosity distance in units set by self.dist_unit (default Mpc)                                                                           
        '''
        if w0!=-1:
            cosmo=FlatwCDM(H0=H0, Om0=Om, w0=w0, Neff=0)
        else:
            cosmo=FlatLambdaCDM(H0=H0, Om0=Om)
            return (cosmo.luminosity_distance(z).to(self.dist_unit).value)*self.Xi(z, Xi0, n)


    def Xi(self, z, Xi0, n):
        return Xi0+(1-Xi0)/(1+z)**n


    ######################
    # SOLVERS FOR DISTANCE-REDSHIFT RELATION
    ######################
    
    def z_from_dLGW_fast(self, r, H0, Om, w0, Xi0, n):
        '''
        Returns redshift for a given luminosity distance r (in Mpc by default). Vectorized
        '''
        if (Om==self.baseValues['Om']) & (w0==-1.):
            dLGrid = self.dLGridGlobals/H0*self.baseValues['H0']
        else:
            if w0==-1:
                cosmo = FlatLambdaCDM(H0=H0, Om0=Om, Neff=0)
            else:
                cosmo = FlatwCDM(H0=H0, Om0=Om, w0=w0, Neff=0)
            dLGrid = cosmo.luminosity_distance(self.zGridGlobals).to(self.dist_unit).value
        z2dL = interpolate.interp1d( dLGrid*self.Xi(self.zGridGlobals, Xi0, n), self.zGridGlobals, kind='cubic', bounds_error=False, fill_value=(0,np.NaN), assume_sorted=True)
        return z2dL(r)

    def z_from_dLGW(self, dL_GW_val, H0, Om, w0, Xi0, n):
        '''Returns redshift for a given luminosity distance dL_GW_val (in Mpc by default)                                         '''
        func = lambda z : self.dLGW(z, H0, Om, w0, Xi0, n) - dL_GW_val
        z = fsolve(func, 0.5)
        return z[0]
    