#!/usr/bin/env python3
#    Copyright (c) 2021 Michele Mancarella <michele.mancarella@unige.ch>
#
#    All rights reserved. Use of this source code is governed by a modified BSD
#    license that can be found in the LICENSE file.


from ..ABSpopulation import RateEvolution
import numpy as np
import astropy.units as u


class PowerLawRateEvolution(RateEvolution):
    
    def __init__(self, unit=u.Gpc, normalized=False):
        RateEvolution.__init__(self)
        
        self.params = ['R0', 'lambdaRedshift']
        self.baseValues = {'R0': 20., #60 , # Gpc^-3 yr^-1
                           'lambdaRedshift':1.8 }
        self.n_params = len(self.params)
        
        self.names = { 'R0':r'$R_0$', 
                           'lambdaRedshift':r'$\lambda$',}
        
        self.normalized=normalized
        if normalized :
            print('The rate evolution returned will not include the overall number of events!')
            self._delete_R0()
        else:
            self._set_rate_units(unit)
    
    def _delete_R0(self):
        
        self.params.remove('R0')
        _ = self.baseValues.pop('R0')
        _ = self.names.pop('R0')
        self.n_params -=1
    
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
        
        z=theta_rate
        if not self.normalized:
            R0, lambdaRedshift = lambdaBBHrate
            return np.log(R0)+lambdaRedshift*np.log1p(z)
        else:
            lambdaRedshift = lambdaBBHrate
            return lambdaRedshift*np.log1p(z)
    
    




class AstroPhRateEvolution(RateEvolution):
    
    def __init__(self, unit=u.Gpc, normalized=False):
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
        
        self.normalized=normalized
        if normalized :
            print('The rate evolution returned will not include the overall number of events!')
            self._delete_R0()
        else:
            self._set_rate_units(unit)
    
    def _delete_R0(self):
        
        self.params.remove('R0')
        _ = self.baseValues.pop('R0')
        _ = self.names.pop('R0')
        self.n_params -=1
    
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
        
        z=theta_rate
        if not self.normalized:
            R0, alphaRedshift , betaRedshift, zp = lambdaBBHrate
            return np.log(R0)+np.log(self._C0(alphaRedshift , betaRedshift, zp))+alphaRedshift*np.log1p(z)-np.log(1+((1+z)/(1+zp))**(alphaRedshift+betaRedshift))
        else:
            alphaRedshift , betaRedshift, zp = lambdaBBHrate
            return alphaRedshift*np.log1p(z)-np.log(1+((1+z)/(1+zp))**(alphaRedshift+betaRedshift))#+np.log(self._C0(alphaRedshift , betaRedshift, zp))

    def _C0(self, alphaRedshift , betaRedshift, zp):
        
        return 1+(1+zp)**(-alphaRedshift-betaRedshift)



    