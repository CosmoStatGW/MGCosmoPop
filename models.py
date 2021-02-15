"""
Created on Wed Jan 20 16:38:28 2021

@author: Michi
"""

from config import *
#import utils
from  cosmo import *
import data
import scipy.stats as ss
from getLambda import get_Lambda
from astropy.cosmology import  Planck15, z_at_value
from scipy.integrate import cumtrapz
import astropy.units as u
import numpy as np
from scipy.special import logsumexp
#####################################################
#####################################################

print('Loading data...')
theta = data.load_data(dataset_name)
m1z, m2z, dL = theta
assert (m1z > 0).all()
assert (m2z > 0).all()
assert (dL > 0).all()
theta_sel, weights_sel, N_gen = data.load_injections_data(dataset_name_injections)
log_weights_sel = np.log(weights_sel)
m1z_sel, m2z_sel, dL_sel = theta
Nobs = theta[0].shape[0]
Tobs = 2.5
OrMassPrior =  data.originalMassPrior(m1z, m2z)
OrDistPrior  = data.originalDistPrior(dL)
logOrMassPrior =  data.originalMassPrior(m1z, m2z)
logOrDistPrior  = data.originalDistPrior(dL)
print('Done data.')
print('theta shape: %s' % str(theta.shape))
print('We have %s observations' % Nobs)

print('Number of total injections: %s' %N_gen)
print('Number of injections with SNR>8: %s' %weights_sel.shape[0])
zmax=z_at_value(Planck15.luminosity_distance, dL_sel.max()*u.Mpc)
print('Max z of injections: %s' %zmax)

#####################################################
#####################################################



#####################################################

def selectionBias(Lambda, m1, m2, z):
    
    #Lambda = get_Lambda(Lambda_test, Lambda_ntest)
    #H0, Om0, w0, Xi0, n, R0, lambdaRedshift, alpha, beta, ml, sl, mh, sh = Lambda
    
    xx = log_dN_dm1zdm2zddL(Lambda, m1, m2, z) - log_weights_sel
    
    #xx*=CCfast(alpha, beta, ml, sl, mh, sh)
    
    logMu = logsumexp(xx) - N_gen
    muSq = np.exp(2*logMu)
    logs2 = logsumexp(xx*xx) -2*N_gen
    SigmaSq = np.exp(logs2) - muSq / N_gen
    Neff = muSq / sigmaSq
    if Neff < 4 * Nobs:
        print('NEED MORE SAMPLES FOR SELECTION EFFECTS! Values of lambda_test: %s' %str(Lambda_test))
    return logMu, Neff


def logLik(Lambda, m1, m2, z):
    """
    Lambda:
     H0, Xi0, n, gamma, alpha, beta, ml, sl, mh, sh
    
    Returns log likelihood for all data
    """
    #Lambda = get_Lambda(Lambda_test, Lambda_ntest)
    lik = log_dN_dm1zdm2zddL(Lambda, m1, m2, z)
    lik -= logOrMassPrior
    lik -= logOrDistPrior
    Nsamples=np.count_nonzero(lik, axis=-1)
    #return np.log(lik.mean(axis=1)) .sum(axis=(-1))
    allLogLiks = logsumexp(lik, axis=-1)-Nsamples
    return (allLogLiks).sum()


def log_prior(Lambda_test, priorLimits):
    if np.isscalar(Lambda_test):
        limInf, limSup = priorLimits[0]
        condition = limInf < Lambda_test < limSup
    else:
        condition = True
        for i, (limInf, limSup) in enumerate(priorLimits):
            condition &= limInf < Lambda_test[i] < limSup

    if condition:
        return 0.0
    return -np.inf




def log_posterior(Lambda_test, Lambda_ntest, priorLimits):
    
    lp = log_prior(Lambda_test, priorLimits)
    if not np.isfinite(lp):
        return -np.inf
    
    Lambda = get_Lambda(Lambda_test, Lambda_ntest)
    
    # Compute source frame masses and redshifts
    precomputed = run_precompute(Lambda)
    
    logPost= logLik(Lambda, precomputed['source_frame_mass1_observations'],precomputed['source_frame_mass2_observations'],precomputed['z_observations'] )+lp
    
    ### Selection bias
    logMu, Neff = selectionBias(Lambda, precomputed['source_frame_mass1_injections'], precomputed['source_frame_mass2_injections'], precomputed['z_injections'] )
    
    ## Effects of uncertainty on selection effect and/or marginalisation over total rate
    ## See 1904.10879
    
    if marginalise_rate:
        logPost -= Nobs*logMu
        if selection_integral_uncertainty:
            logPost+=(3 * Nobs + Nobs * Nobs) / (2 * Neff)
    else:
        #Lambda = get_Lambda(Lambda_test, Lambda_ntest) 
        H0, Om0, w0, Xi0, n, logR0, lambdaRedshift, alpha, beta, ml, sl, mh, sh = Lambda 
        logPost+= Nobs*logR0 #np.log(R0)
        mu=np.exp(logMu)
        R0 = np.exp(logR0)
        logPost -= R0*mu
        if selection_integral_uncertainty:
            logPost+= (R0*mu)*(R0*mu)/ (2 * Neff)
    
    return logPost



#####################################################

def run_precompute(Lambda):
    '''
    Compute only once some quantities that go into selection effects and likelihood
    '''
    H0, Om0, w0, Xi0, n, logR0, lambdaRedshift, alpha, beta, ml, sl, mh, sh = Lambda
    
    precomputed={}
    precomputed['z_observations'] = get_redshift(dL, H0, Om0, w0, Xi0, n)
    precomputed['z_injections'] = get_redshift(dL_sel, H0, Om0, w0, Xi0, n)
    precomputed['source_frame_mass1_observations'] = m1z / (1 + precomputed['z_observations'])
    precomputed['source_frame_mass1_injections'] = m1z_sel / (1 + precomputed['z_injections'])
    
    precomputed['source_frame_mass2_observations'] = m2z / (1 + precomputed['z_observations'])
    precomputed['source_frame_mass2_injections'] = m2z_sel / (1 + precomputed['z_injections'])
    
    # Add normalization of mass function here if needed
    
    return precomputed


def get_redshift(dL, H0, Om0, w0, Xi0, n):
    
    z = z_from_dLGW_fast(dL, H0, Om0, w0, Xi0, n)
    
    if not (z > 0).all():
        print('Parameters H0, Om0, w0, Xi0, n :')
        print(H0, Om0, w0, Xi0, n)
        print('dL = %s' % dL[(z < 0)])
        raise ValueError('negative redshift')
    return z

#####################################################


def dN_dm1dm2dz(Lambda, m1, m2):
    """
    - theta is an array (m1z, m2z, dL) where m1z, m2z, dL are arrays 
    of the GW posterior samples
     
     Lambda = (H0, Xi0, n, lambdaRedshift, lambdaBBH ) 
     lambdaBBH is the parameters of the BBH mass function 
    """
    H0, Om0, w0, Xi0, n, logR0, lambdaRedshift, alpha, beta, ml, sl, mh, sh = Lambda
    lambdaBBH = [alpha, beta, ml, sl, mh, sh]
    return Tobs*dV_dz(z, H0, Om0, w0)*(1 + z)**(lambdaRedshift-1)* massPrior(m1, m2, lambdaBBH)


def dN_dm1zdm2zddL(Lambda, m1, m2, z):
    H0, Om0, w0, Xi0, n, logR0, lambdaRedshift, alpha, beta, ml, sl, mh, sh = Lambda
    #return dN_dm1dm2dz(Lambda, m1, m2) / ( MsourceToMdetJacobian(z) * ddL_dz(z, H0, Om0, w0, Xi0, n) )
    lambdaBBH = [alpha, beta, ml, sl, mh, sh]
    return Tobs*dV_dz(z, H0, Om0, w0)*(1 + z)**(lambdaRedshift-1)* massPrior(m1, m2, lambdaBBH)/  ((1 + z)*(1 + z)) / ddL_dz(z, H0, Om0, w0, Xi0, n) 


def log_dN_dm1zdm2zddL(Lambda, m1, m2, z):
    H0, Om0, w0, Xi0, n, logR0, lambdaRedshift, alpha, beta, ml, sl, mh, sh = Lambda
    #return dN_dm1dm2dz(Lambda, m1, m2) / ( MsourceToMdetJacobian(z) * ddL_dz(z, H0, Om0, w0, Xi0, n) )
    lambdaBBH = [alpha, beta, ml, sl, mh, sh]
    return np.log(Tobs)+log_dV_dz(z, H0, Om0, w0)+np.log(1 + z)*(lambdaRedshift-1)+ logMassPrior(m1, m2, lambdaBBH)-2*np.log(1 + z) - log_ddL_dz(z, H0, Om0, w0, Xi0, n) 


def MsourceToMdetJacobian(z):
    return (1 + z)*(1 + z)



def rateDensityEvol(z, lambdaRedshift):
    """
    merger rate density evolution in redshift (un-normalized)
    """
    return (1 + z)**(lambdaRedshift)




#####################################################

def log_massPrior(m1, m2, lambdaBBH):
    """
    lambdaBBH is the array of parameters of the BBH mass function 
    """
    alpha, beta, ml, sl, mh, sh = lambdaBBH
    #return m1 ** (-alpha) * (m2 / m1) ** beta * f_smooth(m1, ml=ml, sl=sl, mh=mh, sh=sh) * f_smooth(m2, ml=ml, sl=sl, mh=mh, sh=sh) * C1(m1, beta, ml) * C2(alpha, ml, mh)
    return np.log(m1/30)*(-alpha)+np.log(m2/30)*(beta)+np.log(f_smooth(m1, ml=ml, sl=sl, mh=mh, sh=sh)) +np.log(f_smooth(m2, ml=ml, sl=sl, mh=mh, sh=sh)) -np.log(f_smooth(30, ml=ml, sl=sl, mh=mh, sh=sh))*2-2*np.log(30)
    #return (m1)**(-alpha)*(m2)**(beta)*f_smooth(m1, ml=ml, sl=sl, mh=mh, sh=sh)*f_smooth(m2, ml=ml, sl=sl, mh=mh, sh=sh)/CCfast(alpha, beta, ml, sl, mh, sh)



def massPrior(m1, m2, lambdaBBH):
    """
    lambdaBBH is the array of parameters of the BBH mass function 
    """
    alpha, beta, ml, sl, mh, sh = lambdaBBH
    #return m1 ** (-alpha) * (m2 / m1) ** beta * f_smooth(m1, ml=ml, sl=sl, mh=mh, sh=sh) * f_smooth(m2, ml=ml, sl=sl, mh=mh, sh=sh) * C1(m1, beta, ml) * C2(alpha, ml, mh)
    return (m1/30)**(-alpha)*(m2/30)**(beta)*f_smooth(m1, ml=ml, sl=sl, mh=mh, sh=sh)*f_smooth(m2, ml=ml, sl=sl, mh=mh, sh=sh) /f_smooth(30, ml=ml, sl=sl, mh=mh, sh=sh)**2/(30**2)
    #return (m1)**(-alpha)*(m2)**(beta)*f_smooth(m1, ml=ml, sl=sl, mh=mh, sh=sh)*f_smooth(m2, ml=ml, sl=sl, mh=mh, sh=sh)/CCfast(alpha, beta, ml, sl, mh, sh)

def C1(m, beta, ml):
    if beta != -1:
        return (1 + beta) / m / (1 - (ml / m) ** (1 + beta))
    return 1 / m / np.log(m / ml)


def C2(alpha, ml, mh):
    if alpha != 1:
        return (1 - alpha) / (mh ** (1 - alpha) - ml ** (1 - alpha))
    return 1 / np.log(mh / ml)


def f_smooth(m, ml=5, sl=0.1, mh=45, sh=0.1):
    return ss.norm().cdf((np.log(m) - np.log(ml)) / sl) * (1 - ss.norm().cdf((np.log(m) - np.log(mh)) / sh))


def CCfast(alpha=0.75, beta=0, ml=5, sl=0.1, mh=45, sh=0.1):
  
    if (beta!=-1) & (alpha!=1) : 
        return (( mh**(-alpha+beta+2)- ml**(-alpha+beta+2) )/(-alpha+beta+2) - ml**(beta+1) *(mh**(-alpha+1)-ml**(-alpha+1))/(-alpha+1) )/(beta+1)
    else:
        raise ValueError
