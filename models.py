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

#####################################################
#####################################################

print('Loading data...')
theta = data.load_data(dataset_name)
m1z, m2z, dL = theta
assert (m1z > 0).all()
assert (m2z > 0).all()
assert (dL > 0).all()
theta_sel, weights_sel, N_gen = data.load_injections_data(dataset_name_injections)
Nobs = theta[0].shape[0]
Tobs = 2.5
print('Done data.')
print('theta shape: %s' % str(theta.shape))
print('We have %s observations' % Nobs)

print('Number of total injections: %s' %N_gen)
print('Number of injections with SNR>8: %s' %weights_sel.shape[0])
zmax=z_at_value(Planck15.luminosity_distance, theta_sel[2].max()*u.Mpc)
print('Max z of injections: %s' %zmax)

#####################################################
#####################################################



#####################################################

def selectionBias(Lambda_test, Lambda_ntest):
    
    Lambda = get_Lambda(Lambda_test, Lambda_ntest)
    #H0, Om0, w0, Xi0, n, R0, lambdaRedshift, alpha, beta, ml, sl, mh, sh = Lambda
    
    xx = dN_dm1zdm2zddL(Lambda, theta_sel) / weights_sel
    
    #xx*=CCfast(alpha, beta, ml, sl, mh, sh)
    
    mu = np.sum(xx) / N_gen
    s2 = np.sum(xx * xx) /N_gen**2
    sigmaSq = s2 - mu * mu / N_gen
    Neff = mu * mu / sigmaSq
    if Neff < 4 * Nobs:
        print('NEED MORE SAMPLES FOR SELECTION EFFECTS! Values of lambda_test: %s' %str(Lambda_test))
    return (mu, Neff)


def logLik(Lambda_test, Lambda_ntest):
    """
    Lambda:
     H0, Xi0, n, gamma, alpha, beta, ml, sl, mh, sh
    
    Returns log likelihood for all data
    """
    Lambda = get_Lambda(Lambda_test, Lambda_ntest)
    lik = dN_dm1zdm2zddL(Lambda, theta)
    lik /= data.originalMassPrior(m1z, m2z)
    lik /= data.originalDistPrior(dL)
    return np.log(lik.mean(axis=1)).sum(axis=(-1))


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
    
    logPost =  logLik(Lambda_test, Lambda_ntest)+lp 
    
    #### Selection bias  
    mu, Neff = selectionBias(Lambda_test, Lambda_ntest)
    
    ## Effects of uncertainty on selection effect and/or marginalisation over total rate
    ## See 1904.10879
    if marginalise_rate:
        logPost -= Nobs*np.log(mu)
        if selection_integral_uncertainty:
            logPost+=(3 * Nobs + Nobs ** 2) / (2 * Neff)
    else:
        Lambda = get_Lambda(Lambda_test, Lambda_ntest) 
        H0, Om0, w0, Xi0, n, R0, lambdaRedshift, alpha, beta, ml, sl, mh, sh = Lambda 
        logPost+= Nobs*np.log(R0)
        logPost-= R0*mu
        if selection_integral_uncertainty:
            logPost+= (R0*mu)**2/ (2 * Neff)
        
    
    return logPost




#####################################################


def dN_dm1dm2dz(z, Lambda, theta):
    """
    - theta is an array (m1z, m2z, dL) where m1z, m2z, dL are arrays 
    of the GW posterior samples
     
     Lambda = (H0, Xi0, n, lambdaRedshift, lambdaBBH ) 
     lambdaBBH is the parameters of the BBH mass function 
    """
    m1z, m2z, dL = theta
    H0, Om0, w0, Xi0, n, R0, lambdaRedshift, alpha, beta, ml, sl, mh, sh = Lambda
    lambdaBBH = [alpha, beta, ml, sl, mh, sh]
    m1, m2 = m1z / (1 + z), m2z / (1 + z)
    return Tobs*(1 + z)**(-1)*dV_dz(z, H0, Om0, w0)*rateDensityEvol(z, lambdaRedshift) * massPrior(m1, m2, lambdaBBH)


def dN_dm1zdm2zddL(Lambda, theta):
    m1z, m2z, dL = theta
    H0, Om0, w0, Xi0, n, R0, lambdaRedshift, alpha, beta, ml, sl, mh, sh = Lambda
    z = z_from_dLGW_fast(dL, H0, Om0, w0, Xi0, n)
    #if not (z > 0).all():
    #    print('Parameters H0, Xi0, n, R0, lambdaRedshift,  alpha, beta, ml, sl, mh, sh :')
    #    print(H0, Xi0, n, R0, lambdaRedshift, alpha, beta, ml, sl, mh, sh)
    #    print('dL = %s' % dL[(z < 0)])
    #    raise ValueError('negative redshift')
    return dN_dm1dm2dz(z, Lambda, theta) / (MsourceToMdetJacobian(z) * ddL_dz(z, H0, Om0, w0, Xi0, n))


def MsourceToMdetJacobian(z):
    return (1 + z) ** 2



def rateDensityEvol(z, lambdaRedshift):
    """
    merger rate density evolution in redshift (un-normalized)
    """
    return (1 + z)**(lambdaRedshift)




#####################################################

def massPrior(m1, m2, lambdaBBH):
    """
    lambdaBBH is the array of parameters of the BBH mass function 
    """
    alpha, beta, ml, sl, mh, sh = lambdaBBH
    #return m1 ** (-alpha) * (m2 / m1) ** beta * f_smooth(m1, ml=ml, sl=sl, mh=mh, sh=sh) * f_smooth(m2, ml=ml, sl=sl, mh=mh, sh=sh) * C1(m1, beta, ml) * C2(alpha, ml, mh)
    #return (m1/30)**(-alpha)*(m2/30)**(beta)*f_smooth(m1, ml=ml, sl=sl, mh=mh, sh=sh)*f_smooth(m2, ml=ml, sl=sl, mh=mh, sh=sh) /f_smooth(30, ml=ml, sl=sl, mh=mh, sh=sh)**2#/(30**2)
    return (m1)**(-alpha)*(m2)**(beta)*f_smooth(m1, ml=ml, sl=sl, mh=mh, sh=sh)*f_smooth(m2, ml=ml, sl=sl, mh=mh, sh=sh)/CCfast(alpha, beta, ml, sl, mh, sh)

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
