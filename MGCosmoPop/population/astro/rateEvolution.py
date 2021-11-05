#!/usr/bin/env python3
#    Copyright (c) 2021 Michele Mancarella <michele.mancarella@unige.ch>
#
#    All rights reserved. Use of this source code is governed by a modified BSD
#    license that can be found in the LICENSE file.


from ..ABSpopulation import RateEvolution
import numpy as np
import astropy.units as u


class PowerLawRateEvolution(RateEvolution):
    
    def __init__(self, unit=u.Gpc):
        RateEvolution.__init__(self)
        
        self.params = ['R0', 'lambdaRedshift']
        self.baseValues = {'R0': 20., #60 , # Gpc^-3 yr^-1
                           'lambdaRedshift':1.8 }
        self.n_params = len(self.params)
        
        self.names = { 'R0':r'$R_0$', 
                           'lambdaRedshift':r'$\lambda$',}
        
        self._set_rate_units(unit)
    
    def _set_rate_units(self, unit):
        if unit==u.Mpc:
            self.baseValues['R0']*=1e-09
            print('New fiducial value for R0 using yr^-1 Mpc^-3: %s' %self.baseValues['R0'])
        elif unit==u.Gpc:
            print('Rate R0 is in yr^-1 Gpc^-3')
        else:
            raise ValueError
    
    def log_dNdVdt(self, theta_rate, lambdaBBHrate):
        '''
        Evolution of total rate with redshift: R0*(1+z)**lambda
        '''
        R0, lambdaRedshift = lambdaBBHrate
        z=theta_rate
        return np.log(R0)+lambdaRedshift*np.log1p(z)
    
    




class AstroPhRateEvolution(RateEvolution):
    
    def __init__(self, unit=u.Gpc):
        RateEvolution.__init__(self)
        
        self.params = ['R0', 'alphaRedshift', 'betaRedshift', 'zp']
        self.baseValues = {'R0': 20., #60 , # Gpc^-3 yr^-1
                           'alphaRedshift':1.1,
                           'betaRedshift': 2.6,
                           'zp': 2.4}
        
        self.n_params = len(self.params)
        
        self.names = { 'R0':r'$R_0$', 
                           'alphaRedshift':r'$\alpha_z$',
                           'betaRedshift':r'$\beta_z$',
                           'zp': r'$z_p$'}
        
        self._set_rate_units(unit)
    
    def _set_rate_units(self, unit):
        if unit==u.Mpc:
            self.baseValues['R0']*=1e-09
            print('New fiducial value for R0 using yr^-1 Mpc^-3: %s' %self.baseValues['R0'])
        elif unit==u.Gpc:
            print('Rate R0 is in yr^-1 Gpc^-3')
        else:
            raise ValueError
    
    def log_dNdVdt(self, theta_rate, lambdaBBHrate):
        '''
        Evolution of total rate with redshift: R0*(1+z)**lambda
        '''
        R0, alphaRedshift , betaRedshift, zp = lambdaBBHrate
        z=theta_rate
        return np.log(R0)+np.log(self._C0(alphaRedshift , betaRedshift, zp))+alphaRedshift*np.log1p(z)-np.log(1+((1+z)/(1+zp))**(alphaRedshift+betaRedshift))


    def _C0(self, alphaRedshift , betaRedshift, zp):
        
        return 1+(1+zp)**(-alphaRedshift-betaRedshift)




class DummyRateEvolution(RateEvolution):
    
    def __init__(self):
        RateEvolution.__init__(self)
        
        self.params = ['R1', 'gamma']
        self.baseValues = {'R1': 1, #60 , # Gpc^-3 yr^-1
                           'gamma': -1 ,}
        self.n_params = len(self.params)
        
        self.names = {
                           'R1':r'$R_1$', 
                           'gamma':r'$\gamma$',}
        

    def log_dNdVdt(self, theta_rate, lambdaBBHrate):
        '''
        Evolution of total rate with redshift: R0*(1+z)**lambda
        '''
        R1, gamma = lambdaBBHrate
        z=theta_rate
        return np.log(R1)+gamma*np.log1p(z)
    
    