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
    
    def __init__(self, unit=u.Gpc, normalized=False, zmax=20):
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
        self.zmax = zmax
        
        print('The rate will be zero after z=%s'%self.zmax)
        
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
            pz =  np.log(R0)+np.log(self._C0(alphaRedshift , betaRedshift, zp))+alphaRedshift*np.log1p(z)-np.log(1+((1+z)/(1+zp))**(alphaRedshift+betaRedshift))
        else:
            alphaRedshift , betaRedshift, zp = lambdaBBHrate
            pz =  alphaRedshift*np.log1p(z)-np.log(1+((1+z)/(1+zp))**(alphaRedshift+betaRedshift))#+np.log(self._C0(alphaRedshift , betaRedshift, zp))
            
        return np.where( z<self.zmax, pz, np.NINF )

    def _C0(self, alphaRedshift , betaRedshift, zp):
        
        return 1+(1+zp)**(-alphaRedshift-betaRedshift)
    
    
    
    
class RateEvolutionCOBA(RateEvolution):
        
        def __init__(self, unit=u.Gpc, normalized=False):
            RateEvolution.__init__(self)
            
            self.params = ['R0', 'alphaRedshift', 'betaRedshift', 'zp', 'zscale', 'gamma', 'zmax', 'dz']
            self.baseValues = {'R0': 3.187829599208797e-08*1e09, #60 , # Gpc^-3 yr^-1
                               'alphaRedshift':1.1,
                               'betaRedshift': 2.6,
                               'zp': 2.4, 
                               'zscale':8.22, 'gamma':5.05, 'zmax':9.38, 'dz':5.62
                               
                               }
            
            
            self.n_params = len(self.params)
            
            self.names = { 'R0':r'$R_0$', 
                               'alphaRedshift':r'$\alpha_z$',
                               'betaRedshift':r'$\beta_z$',
                               'zp': r'$z_p$' ,'zscale':r'$z_{\rm scale}$', 'gamma':r'$\gamma$', 'zmax':r'$z_{\rm max}$', 'dz':r'$dz$'}
            
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
                R0, alphaRedshift , betaRedshift, zp, zscale, gamma, zmax, dz = lambdaBBHrate
                logN0 = np.log(R0)+np.log(self._C0(alphaRedshift , betaRedshift, zp))
            else:
                alphaRedshift , betaRedshift, zp, zscale, gamma, zmax, dz = lambdaBBHrate
                logN0 = 1.  #np.log(self._C0(alphaRedshift , betaRedshift, zp))
                
            res=np.zeros(z.shape)
            where_correct = z>zscale
            
            res[~where_correct] = self._log_MD_profile(z[~where_correct], alphaRedshift , betaRedshift,zp, logN0)
            
            MDVal = np.exp(self._log_MD_profile(zscale, alphaRedshift , betaRedshift,zp, logN0)) #MDprofile(zscale,alpha,beta,zp,N0)
            NewVal = np.exp(self._log_new_profile( zscale, gamma, logN0)) #NewProfile(zscale,gam,N0)
            C0 = MDVal - NewVal 
                
            res[where_correct] = np.log( C0 + np.exp( self._log_new_profile( z[where_correct], gamma, logN0) ) ) +np.log(self._filterFun( z[where_correct], zmax,dz))
                
            return res

        
        def _log_new_profile(self, z, gam, logN0 ):
            return logN0+gam*np.log1p(z) #N0*(1.+z)**gam
        
        
        def _log_MD_profile(self, z, alpha,beta,zp, logN0):
            return logN0+alpha*np.log1p(z)-np.log(1+((1+z)/(1+zp))**(alpha+beta))
            
        
        def _C0(self, alphaRedshift , betaRedshift, zp):
            
            return 1+(1+zp)**(-alphaRedshift-betaRedshift)
        
        
        def _filterFun(self, z, zmax,dz):
            return np.where(z<zmax, 1., np.where(z<zmax+dz, 1./(1+np.exp(-dz/(-zmax+z) - dz/((-zmax+z)-dz))), 0.))



    