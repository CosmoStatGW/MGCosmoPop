#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 16:38:28 2021

@author: Michi
"""


####################
# This module contains modeling of the BBH mass function and rate
####################


from utils import *
from dataFarr import *
import scipy.stats as ss




#((20, 140), (0.3, 10), (20, 150)) # (-10, 250), (-10, 250), (-10, 250), (-10, 250), (-10, 250), (-10, 250))



###########################################################################
print('Loading data...')
theta = load_mock_data()
m1z, m2z, dL = theta
assert (m1z > 0).all()
assert (m2z > 0).all()
assert (dL > 0).all()
theta_sel, weights_sel, N_gen = load_injections_data()
print('Done data.')
print('theta shape: %s' %str(theta.shape))
print('We have %s observations' %theta.shape[1])

###########################################################################

def get_Lambda(Lambda_test, Lambda_ntest):
    H0, Xi0, mh = Lambda_test
    n, lambdaRedshift, alpha, beta, ml, sl, sh  = Lambda_ntest
    Lambda = [H0, Xi0, n, lambdaRedshift,  alpha, beta, ml, sl, mh, sh]
    return Lambda


def alphabias(Lambda_test, Lambda_ntest):
    Lambda = get_Lambda(Lambda_test, Lambda_ntest)
    return np.sum(dN_dm1zdm2zddL( Lambda)/weights_sel)/N_gen


def logLik(Lambda_test, Lambda_ntest):
    '''
    Lambda:
     H0, Xi0, n, gamma, alpha, beta, ml, sl, mh, sh
    
    Returns log likelihood for all data
    '''
    #H0, Xi0,  mh = Lambda_test
    #n, lambdaRedshift, alpha, beta, ml, sl, sh  = Lambda_ntest
    #Lambda = [H0, Xi0, n, lambdaRedshift,  alpha, beta, ml, sl, mh, sh]
    Lambda = get_Lambda(Lambda_test, Lambda_ntest)
    #m1z, m2z, dL = theta
    lik = dN_dm1zdm2zddL(Lambda) # ( n_obs x n_samples ) 
    
    lik/=originalMassPrior(m1z, m2z)
    lik/=originalDistPrior(dL)
        
    return np.log(lik.mean(axis=1)).sum(axis=-1)

#### The mean is taken over axis = 1 (instead of -1)

def log_prior(Lambda_test, priorLimits):
    #H0, Xi0, mh = Lambda_test
    #allVariables = flatten2([Lambda_test,])
    condition=True
    for i, (limInf, limSup) in enumerate(priorLimits):
        condition &= limInf<Lambda_test[i]<limSup 
    if condition: 
        return 0.0
    else:
        return -np.inf


def log_posterior(Lambda_test, Lambda_ntest, priorLimits):
    lp = log_prior(Lambda_test, priorLimits)
    Nobs=theta[0].shape[0]
    if not np.isfinite(lp):
       return -np.inf
    return logLik(Lambda_test, Lambda_ntest)-Nobs*np.log(alphabias(Lambda_test, Lambda_ntest)) + lp


###########################################################################


def originalMassPrior(m1z, m2z):
    return np.ones(m1z.shape)

def originalDistPrior(dL):
    return np.ones(dL.shape)



def dN_dm1dm2dz(z,  Lambda) :
    '''
    - theta is an array (m1z, m2z, dL) where m1z, m2z, dL are arrays 
    of the GW posterior samples
     
     Lambda = (H0, Xi0, n, lambdaRedshift, lambdaBBH ) 
     lambdaBBH is the parameters of the BBH mass function 
    '''
    
    #m1z, m2z, dL = theta
    H0, Xi0, n, lambdaRedshift,  alpha, beta, ml, sl, mh, sh  = Lambda
    lambdaBBH=[ alpha, beta, ml, sl, mh, sh]
    
    m1, m2 = m1z/(1+z), m2z/(1+z)
    #assert (m1 > 0).all()
    #assert (m2 > 0).all()
    return redshiftPrior(z, lambdaRedshift, H0)*massPrior(m1, m2, lambdaBBH)



def dN_dm1zdm2zddL(Lambda) :
    
    #m1z, m2z, dL = theta
    H0, Xi0, n, lambdaRedshift,  alpha, beta, ml, sl, mh, sh  = Lambda
    
    z=z_from_dLGW_fast(dL, H0, Xi0, n)
    if not (z > 0).all():
        print('Parameters H0, Xi0, n, lambdaRedshift,  alpha, beta, ml, sl, mh, sh :')
        print(H0, Xi0, n, lambdaRedshift,  alpha, beta, ml, sl, mh, sh)
        
        print('dL = %s' %dL[z<0])
        raise ValueError('negative redshift')

    return  dN_dm1dm2dz(z,  Lambda)/(redshiftJacobian(z)*ddL_dz(z, H0, Xi0, n))



def redshiftJacobian(z):
    return (1+z)**2



def redshiftPrior(z, gamma,H0):
    '''
    dV/dz *(1+z)^(gamma-1)  [Mpc^3]
    '''
    return 4*np.pi*(1+z)**(gamma-1)*((clight/H0)**3)*j(z) 
### Depends on H0 in the arguments


def massPrior(m1, m2, lambdaBBH):
    '''
    lambdaBBH is the array of parameters of the BBH mass function 
    '''
    alpha, beta, ml, sl, mh, sh = lambdaBBH
    return (m1/30)**(-alpha)*(m2/30)**(beta)*f_smooth(m1, ml=ml, sl=sl, mh=mh, sh=sh)*f_smooth(m2, ml=ml, sl=sl, mh=mh, sh=sh)  
  ### gamma isn't in lambdaBBH
    
def f_smooth(m, ml=5, sl=0.1, mh=45, sh=0.1):
    return ss.norm().cdf((np.log(m)-np.log(ml))/sl)*(1-ss.norm().cdf((np.log(m)-np.log(mh))/sh))
## The last sigma is sh, not sl
