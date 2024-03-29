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
from scipy.interpolate import interp2d

import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
#sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))


class Cosmo(object):
    
    def __init__(self, dist_unit = u.Gpc, baseValues=None, use_interpolators=False):
        
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
        
        if self.baseValues['w0']==-1:
            self.cosmoGlobals = FlatwCDM( H0=self.baseValues['H0'], Om0=self.baseValues['Om'], w0=self.baseValues['w0'])
        else:
            self.cosmoGlobals = FlatLambdaCDM( H0=self.baseValues['H0'], Om0=self.baseValues['Om'],)
        self.zGridGlobals = np.concatenate([ np.logspace(start=-15, stop=np.log10(9.99e-09), base=10, num=10), np.logspace(start=-8, stop=np.log10(7.99), base=10, num=1000), np.logspace(start=np.log10(8), stop=20, base=10, num=100)])
        self.dLGridGlobals = self.cosmoGlobals.luminosity_distance(self.zGridGlobals).to(dist_unit).value

        self.clight=const.c.value*1e-03 # c in km/s
        
        # make interpolators
        if use_interpolators:
            try:
                base_dir = SCRIPT_DIR #os.getcwd()
                zgrid = np.load(base_dir+'/zgrid_uu.npy')
                Omgrid = np.load(base_dir+'/Omgrid_uu.npy')
                uuvals = np.load(base_dir+'/vals_grid_uu.npy')
                self.uu_interp = interp2d(zgrid, Omgrid, uuvals.T, kind='cubic', copy=True, bounds_error=False, fill_value=None)
                print('Built interpolator for dimensioneless comoving distance in LCDM')
                #Evals = np.load(base_dir+'/vals_grid_E.npy')
                #self.E_interp = interp2d(zgrid, Omgrid, Evals.T, kind='cubic', copy=True, bounds_error=False, fill_value=None, assume_sorted=False)
                #print('Built interpolator for E(z) in LCDM')
                self.use_interpolators=True
            except Exception as e:
                print(e)
                print('No data for interpolators found for dimensioneless comoving distance and E(z) !')
                self.use_interpolators=False
                pass
        else:
            self.use_interpolators=False
    
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
            return 70/self.clight*FlatwCDM(H0=70, Om0=Om, w0=w0).comoving_distance(z).to(u.Mpc).value
        else:
            if self.use_interpolators:
                or_shape = z.shape
                myzs = z.flatten()
                idxs_sort = np.argsort(myzs)
                idxs_sort_inv = np.empty_like(idxs_sort)
                idxs_sort_inv[idxs_sort] = np.arange(idxs_sort.size)
                res = self.uu_interp(myzs[idxs_sort], Om, assume_sorted=True)[idxs_sort_inv].reshape(or_shape)
                return res
            else:
                return 70/self.clight*FlatLambdaCDM(H0=70, Om0=Om).comoving_distance(z).to(u.Mpc).value

    def E(self,z, Om, w0):
        '''
        E(z). Does not depend on H0
        '''
        if w0!=-1:
            return FlatwCDM(H0=70, Om0=Om, w0=w0).efunc(z)
        else:
            #if self.use_interpolators:
            #    return self.E_interp(z, Om, assume_sorted=True)
            #else:
            return np.sqrt( Om*(1+z)**3+(1-Om) )#FlatLambdaCDM(H0=70, Om0=Om).efunc(z)



    def dV_dz(self,z, H0, Om, w0):
        '''
        Jacobian of comoving volume, with correct dimensions [Mpc^3]. Depends on H0
        '''
        if w0!=-1:
            return 4*np.pi*FlatwCDM(H0=H0, Om0=Om, w0=w0).differential_comoving_volume(z).to(self.dist_unit**3/u.sr).value
        else:
            return 4*np.pi*FlatLambdaCDM(H0=H0, Om0=Om,).differential_comoving_volume(z).to(self.dist_unit**3/u.sr).value

    def log_dV_dz(self, z, H0, Om0, w0):
        res =  np.log(4*np.pi)+3*np.log(self.clight)-3*np.log(H0)+2*np.log(self.uu(z, Om0, w0))-np.log(self.E(z, Om0, w0))
        if self.dist_unit==u.Gpc:
            res -=9*np.log(10)
        return res
    
    def log_j(self, z, Om0, w0):
        res =  np.log(4*np.pi)+2*np.log(self.uu(z, Om0, w0))-np.log(self.E(z, Om0, w0))
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


    def log_ddL_dz(self, z, H0, Om0, w0, Xi0, n, dL=None):
        if self.dist_unit==u.Gpc: # and dL is not None:
            H0*=1e03
        
        if Xi0!=1 and n!=0:
        
            if dL is None:
                res =  np.log(self.clight)-np.log(H0)+np.log(self.sPrime(z, Xi0, n)*self.uu(z, Om0, w0)+self.s(z, Xi0, n)/self.E(z, Om0, w0))
        
            else:
                #res = np.log(  dL*( 1-(n*(1-Xi0))/(Xi0*(1+z)**n+1-Xi0))/(1+z) + self.clight*(1+z)/(H0*self.E(z, Om0, w0)) )
                res = np.log( dL/(1+z)*( 1-(n*(1-Xi0))/(self.Xi(z, Xi0, n)*(1+z)**n) )+self.clight*(1+z)*self.Xi(z, Xi0, n)/(H0*self.E(z, Om0, w0)))
        else:
#            print('Using GR expression')
            if dL is None:
                res = np.log(self.clight)-np.log(H0)+np.log( self.uu(z, Om0, w0) +(1+z)/(self.E(z, Om0, w0)) )
            else:
                res = np.log( dL/(1+z) + self.clight*(1+z)/(H0*self.E(z, Om0, w0)) )
        
        #if self.dist_unit==u.Gpc and dL is None:
        #    res -= 3*np.log(10)
        return res

    def dLGW(self, z, H0, Om, w0, Xi0, n):
        '''                                                                                                          
        Modified GW luminosity distance in units set by self.dist_unit (default Mpc)                                                                           
        '''
        if w0!=-1:
            cosmo=FlatwCDM(H0=H0, Om0=Om, w0=w0, )
        else:
            cosmo=FlatLambdaCDM(H0=H0, Om0=Om)
        if Xi0!=1 and n!=0:
            return (cosmo.luminosity_distance(z).to(self.dist_unit).value)*self.Xi(z, Xi0, n)
        else:
            return cosmo.luminosity_distance(z).to(self.dist_unit).value

    def Xi(self, z, Xi0, n):
        if Xi0!=1 and n!=0:
            Xi=Xi0+(1-Xi0)/(1+z)**n
        else: 
            Xi=1.
        return Xi


    ######################
    # SOLVERS FOR DISTANCE-REDSHIFT RELATION
    ######################
    
    def z_from_dLGW_fast(self, r, H0, Om, w0, Xi0, n):
        '''
        Returns redshift for a given luminosity distance r (in Mpc by default). Vectorized
        '''
        #print('z_from_dLGW_fast call' )
        if False: #(Om==self.baseValues['Om']) & (w0==-1.):
            dLGrid = self.dLGridGlobals/H0*self.baseValues['H0']
        else:
            if w0==-1:
                cosmo = FlatLambdaCDM(H0=H0, Om0=Om, )
            else:
                cosmo = FlatwCDM(H0=H0, Om0=Om, w0=w0, )
            dLGrid = cosmo.luminosity_distance(self.zGridGlobals).to(self.dist_unit).value
        z2dL = interpolate.interp1d( dLGrid*self.Xi(self.zGridGlobals, Xi0, n), self.zGridGlobals, kind='cubic', bounds_error=False, fill_value=(0,np.NaN), assume_sorted=False)
        #print( z2dL(1) )
        return z2dL(r)

    def z_from_dLGW(self, dL_GW_val, H0, Om, w0, Xi0, n):
        '''Returns redshift for a given luminosity distance dL_GW_val (in Mpc by default)                                         '''
        func = lambda z : self.dLGW(z, H0, Om, w0, Xi0, n) - dL_GW_val
        z = fsolve(func, 0.5)
        return z[0]
    
