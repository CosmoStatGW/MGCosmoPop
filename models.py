"""
Created on Wed Jan 20 16:38:28 2021

@author: Michi
"""

#from Globals import *
import config
#from config import *
#import utils
#from  cosmo import *
import cosmo
import data
import scipy.stats as ss
from getLambda import get_Lambda
from astropy.cosmology import  Planck15, z_at_value
#from scipy.integrate import cumtrapz
import astropy.units as u
import numpy as np
from scipy.special import logsumexp
#####################################################
#####################################################

print('Loading data...')
theta, Nsamples = data.load_data(config.dataset_name)
Nobs = theta[0].shape[0]
logNobs = np.log(Nobs)
m1z, m2z, dL = theta
assert (m1z > 0).all()
assert (m2z > 0).all()
assert (dL > 0).all()
Tobs = 2.5

print('theta shape: %s' % str(theta.shape))
print('We have %s observations' % Nobs)

print('Loading injections...')
theta_sel, weights_sel, N_gen = data.load_injections_data(config.dataset_name_injections)
log_weights_sel = np.log(weights_sel)
m1z_sel, m2z_sel, dL_sel = theta_sel
logN_gen=np.log(N_gen)
print('Number of total injections: %s' %N_gen)
print('Number of injections with SNR>8: %s' %weights_sel.shape[0])
zmax=z_at_value(Planck15.luminosity_distance, dL_sel.max()*u.Mpc)
print('Max z of injections: %s' %zmax)



OrMassPrior =  data.originalMassPrior(m1z, m2z)
OrDistPrior  = data.originalDistPrior(dL)
logOrMassPrior =  data.originalMassPrior(m1z, m2z)
logOrDistPrior  = data.originalDistPrior(dL)



#####################################################
#####################################################



#####################################################

def selectionBias(Lambda, m1, m2, z, get_neff=False, verbose=False):
    
    #m1, m2, z = precomputed['m1'], precomputed['m2'], precomputed['z']
    #Lambda = get_Lambda(Lambda_test, Lambda_ntest)
    #H0, Om0, w0, Xi0, n, R0, lambdaRedshift, alpha, beta, ml, sl, mh, sh = Lambda
    
    xx = dN_dm1zdm2zddL(Lambda, m1, m2, z)/weights_sel
        
    mu = xx.sum()/N_gen
    if not get_neff:
        return mu, np.NaN #np.repeat(np.NaN, logMu.shape[0] )
    else:
        muSq = mu*mu
        SigmaSq = np.sum(xx*xx)/N_gen**2 - muSq/N_gen
        Neff = muSq/SigmaSq
        if Neff < 4 * Nobs and verbose:
            print('NEED MORE SAMPLES FOR SELECTION EFFECTS! ') #Values of lambda_test: %s' %str(Lambda_test))
        return mu, Neff



def logSelectionBias(Lambda, m1, m2, z, get_neff=False, verbose=False):
    
    #m1, m2, z = precomputed['m1'], precomputed['m2'], precomputed['z']
    #Lambda = get_Lambda(Lambda_test, Lambda_ntest)
    #H0, Om0, w0, Xi0, n, R0, lambdaRedshift, alpha, beta, ml, sl, mh, sh = Lambda
    
    logxx = log_dN_dm1zdm2zddL(Lambda, m1, m2, z) - log_weights_sel   
    logMu = logsumexp(logxx) - logN_gen
    
    if not get_neff:
        return logMu, np.NaN #np.repeat(np.NaN, logMu.shape[0] )
    else:
        muSq = np.exp(2*logMu).astype('float128')
        logs2 = (logsumexp(2*logxx) -2*logN_gen).astype('float128')
        #logSigmaSq = -logN_gen-2*logMu+logsumexp(2*xx)
        SigmaSq = np.exp(logs2) - muSq/N_gen
        Neff = muSq/SigmaSq
        if Neff < 4 * Nobs and verbose:
            print('NEED MORE SAMPLES FOR SELECTION EFFECTS! ')#Values of lambda_test: %s' %str(Lambda_test))
        return logMu, Neff


def logLik(Lambda, m1, m2, z):
    """
    Lambda:
     H0, Xi0, n, gamma, alpha, beta, ml, sl, mh, sh
    
    Returns log likelihood for all data
    """
    #m1, m2, z = precomputed['m1'], precomputed['m2'], precomputed['z']
    #Lambda = get_Lambda(Lambda_test, Lambda_ntest)
    logLik_ = log_dN_dm1zdm2zddL(Lambda, m1, m2, z)
    logLik_ -= logOrMassPrior
    logLik_ -= logOrDistPrior
    #return np.log(lik.mean(axis=1)) .sum(axis=(-1))
    allLogLiks = logsumexp(logLik_, axis=-1)-Nsamples
    return allLogLiks.sum()


def log_prior(Lambda_test, priorLimits, params_inference):
    
    #  = [ (priorLimits.limInf[param], priorLimits.limSup[param] ) for param in priorLimits.names ]
    
    if np.isscalar(Lambda_test):
        limInf = priorLimits.limInf[params_inference]
        limSup = priorLimits.limSup[params_inference]#priorLimits[0]
        condition = limInf < Lambda_test < limSup
    else:
        condition = True
        for i, param in enumerate(params_inference): #(limInf, limSup) in enumerate(priorLimits):
            limInf = priorLimits.limInf[param]
            limSup = priorLimits.limSup[param]
            condition &= limInf < Lambda_test[i] < limSup

    if condition:
        return priorLimits.get_logVals(Lambda_test, params_inference)
    else: 
        return -np.inf




def log_posterior(Lambda_test, Lambda_ntest, priorLimits, verbose_bias, params_inference):
    
    
    
    lp = log_prior(Lambda_test, priorLimits, params_inference)
    if not np.isfinite(lp):
        return -np.inf
    
    Lambda = get_Lambda(Lambda_test, Lambda_ntest)
    # Compute source frame masses and redshifts
    m1_obs, m2_obs, z_obs = get_mass_redshift(Lambda, which_data='obs')
    logPost= logLik(Lambda, m1_obs, m2_obs, z_obs )+ lp
    
    ### Selection bias
    m1_inj, m2_inj, z_inj = get_mass_redshift(Lambda, which_data='inj')
    logMu, Neff = logSelectionBias(Lambda, m1_inj, m2_inj, z_inj, get_neff = config.selection_integral_uncertainty, verbose=verbose_bias )
    
    ## Effects of uncertainty on selection effect and/or marginalisation over total rate
    ## See 1904.10879
    
    if config.marginalise_rate:
        logPost -= Nobs*logMu
        if config.selection_integral_uncertainty:
            logPost+=(3 * Nobs + Nobs * Nobs) / (2 * Neff)
    else:
        #Lambda = get_Lambda(Lambda_test, Lambda_ntest) 
        H0, Om0, w0, Xi0, n, R0, lambdaRedshift, alpha, beta, ml, sl, mh, sh = Lambda 
        logR0 = np.log(R0)
        logPost += Nobs*logR0 #np.log(R0)
        mu=np.exp(logMu)
        #R0 = np.exp(logR0)
        logPost -= R0*mu
        if config.selection_integral_uncertainty:
            logPost+= (R0*mu)*(R0*mu)/ (2 * Neff)
    
    return logPost



#####################################################

def get_mass_redshift(Lambda, which_data):
    '''
    Compute only once some quantities that go into selection effects and likelihood
    '''
    H0, Om0, w0, Xi0, n, R0, lambdaRedshift, alpha, beta, ml, sl, mh, sh = Lambda
    
    if which_data=='obs':      
        z = get_redshift(  dL, H0, Om0, w0, Xi0, n)
        m1 = m1z / (1 + z)    
        m2 = m2z / (1 + z)
    elif which_data=='inj':
        #print('Precomuting with inj data')
        z = get_redshift( dL_sel, H0, Om0, w0, Xi0, n)
        m1 = m1z_sel / (1 + z)    
        m2 = m2z_sel / (1 + z)
        
    return m1, m2, z




def get_redshift(r, H0, Om0, w0, Xi0, n):
    
    z = cosmo.z_from_dLGW_fast(r, H0, Om0, w0, Xi0, n)
    
    if not (z > 0).all():
        print('Parameters H0, Om0, w0, Xi0, n :')
        print(H0, Om0, w0, Xi0, n)
        print('dL = %s' % r[(z < 0)])
        raise ValueError('negative redshift')
    return z

#####################################################


def dN_dm1dm2dz(Lambda, m1, m2, z):
    """
    - theta is an array (m1z, m2z, dL) where m1z, m2z, dL are arrays 
    of the GW posterior samples
     
     Lambda = (H0, Xi0, n, lambdaRedshift, lambdaBBH ) 
     lambdaBBH is the parameters of the BBH mass function 
    """
    H0, Om0, w0, Xi0, n, R0, lambdaRedshift, alpha, beta, ml, sl, mh, sh = Lambda
    lambdaBBH = [alpha, beta, ml, sl, mh, sh]
    return Tobs*cosmo.dV_dz(z, H0, Om0, w0)*(1 + z)**(lambdaRedshift-1)* massPrior(m1, m2, lambdaBBH)


def dN_dm1zdm2zddL(Lambda, m1, m2, z):
    H0, Om0, w0, Xi0, n, R0, lambdaRedshift, alpha, beta, ml, sl, mh, sh = Lambda
    #return dN_dm1dm2dz(Lambda, m1, m2) / ( MsourceToMdetJacobian(z) * ddL_dz(z, H0, Om0, w0, Xi0, n) )
    lambdaBBH = [alpha, beta, ml, sl, mh, sh]
    return Tobs*cosmo.dV_dz(z, H0, Om0, w0)*(1 + z)**(lambdaRedshift-1)* massPrior(m1, m2, lambdaBBH)/  ((1 + z)*(1 + z)) / cosmo.ddL_dz(z, H0, Om0, w0, Xi0, n) 


def log_dN_dm1zdm2zddL(Lambda, m1, m2, z):
    H0, Om0, w0, Xi0, n, R0, lambdaRedshift, alpha, beta, ml, sl, mh, sh = Lambda
    #return dN_dm1dm2dz(Lambda, m1, m2) / ( MsourceToMdetJacobian(z) * ddL_dz(z, H0, Om0, w0, Xi0, n) )
    lambdaBBH = [alpha, beta, ml, sl, mh, sh]
    return np.log(Tobs)+cosmo.log_dV_dz(z, H0, Om0, w0)+np.log(1 + z)*(lambdaRedshift-1)+ logMassPrior(m1, m2, lambdaBBH)-2*np.log(1 + z) - cosmo.log_ddL_dz(z, H0, Om0, w0, Xi0, n) 


def MsourceToMdetJacobian(z):
    return (1 + z)*(1 + z)



def rateDensityEvol(z, lambdaRedshift):
    """
    merger rate density evolution in redshift (un-normalized)
    """
    return (1 + z)**(lambdaRedshift)




#####################################################

def eval_fsmooth(m, ml=5, sl=0.1, mh=45, sh=0.1, nSigma=5):
    
    lowlim = max(0, ml-nSigma*sl)
    
    logPdf = np.zeros_like(m)
    
    support_low = (
        ( lowlim <= m) &
        (ml+nSigma*sl >= m)
    )

    support_high = (
        ( mh-nSigma*sh <= m) &
        (mh+nSigma*sh >= m)
    )

    support = ( (support_low)  |  (support_high) )

    # We evaluate the smooth component of the logPdf only where it has 
    # nontrivial values; elsewhere we leave it = to zero
    if m.ndim==1:
        m = m[support]
        logPdf[support] = logf_smooth(m, ml=ml, sl=sl, mh=mh, sh=sh)-logf_smooth(30, ml=ml, sl=sl, mh=mh, sh=sh)
    else:
        mask = ~support
        # set to nan the points outside the support. 
        # Then, do not evaluate f_smooth there, but leave them to zero
        m = np.ma.array( m,  mask=mask, fill_value=np.NaN).data
        logPdf = np.where( ~np.isnan(m.data), logf_smooth(m.data, ml=ml, sl=sl, mh=mh, sh=sh)-logf_smooth(30, ml=ml, sl=sl, mh=mh, sh=sh)  , 0)
    
    #print('Eval fsmooth shape: %s' %str(logPdf.shape))
    
    return logPdf


def eval_pdf_support(m, ml=5, sl=0.1, mh=45, sh=0.1, nSigma=5):
    lowlim = max(0, ml-nSigma*sl)
    support =  (
        ( lowlim < m) &
        (mh+nSigma*sh > m)
    )    
    #print(' eval_pdf_support shape: %s' %str(support.shape))
    return support
    

def logMassPrior_1(m1, m2, lambdaBBH):
    """
    lambdaBBH is the array of parameters of the BBH mass function 
    """
    alpha, beta, ml, sl, mh, sh = lambdaBBH
    
    support = eval_pdf_support(m1, ml=ml, sl=sl, mh=mh, sh=sh )
    logpdf = np.full( m1.shape, -np.inf)
    # initialize pdf to 0 (-> logpdf to negative infinity)
    #print('logMassPrior initial logpdf shape: %s' %str(logpdf.shape))
    
    if m1.ndim==1:
        
        # Do not evaluate if m1, m2 are outside support
        # 1D version:
        m1, m2 = m1[support], m2[support]
        #print('logMassPrior m1 shape after applying mask: %s' %str(m1.shape))
        #print('logMassPrior logpdf[support] shape : %s' %str(logpdf[support].shape))
        # Check where to evaluate f_smooth and do it ;
        # only evaluate if m is between [ ml-5sl, mh +5sh ]
        m1_smooth = eval_fsmooth(m1, ml=ml, sl=sl, mh=mh, sh=sh )
        m2_smooth = eval_fsmooth(m2, ml=ml, sl=sl, mh=mh, sh=sh )
        #print('logMassPrior m2_smooth shape: %s' %str(m2_smooth.shape))
        # add in the smoothings where needed, and set logpdf to zero (-> pdf to 1)
        # inside the rest of support 
        logpdf[support] = (m1_smooth+m2_smooth)
        
        # add the power law components
        logpdf[support] += ( np.log(m1)*(-alpha)+alpha*np.log(30) )
        logpdf[support] += (  np.log(m2)*(beta) -beta*np.log(30) )
            
        # normalization 
        logpdf[support]-= 2*np.log(30)
    else:
        mask = ~support
        m1 = np.ma.array( m1,  mask=mask, fill_value=-1).data
        m2 = np.ma.array( m2,  mask=mask, fill_value=-1).data
        
        m1_smooth = eval_fsmooth(m1, ml=ml, sl=sl, mh=mh, sh=sh )
        m2_smooth = eval_fsmooth(m2, ml=ml, sl=sl, mh=mh, sh=sh )
        #print('logMassPrior m2_smooth shape: %s' %str(m2_smooth.shape))
        
        powLaw_m2 = np.log(m2)*(beta) -beta*np.log(30)
        powLaw_m1 =np.log(m1)*(-alpha)+alpha*np.log(30)
        
        norm = - 2*np.log(30)
        
        logpdf = np.where(mask, m1_smooth+m2_smooth+powLaw_m2+powLaw_m1+norm,  -np.inf )
        
        
    return logpdf


def logMassPrior(m1, m2, lambdaBBH):
    alpha, beta, ml, sl, mh, sh = lambdaBBH
    return np.log(m1)*(-alpha)+alpha*np.log(30) +np.log(m2)*(beta) -beta*np.log(30) +logf_smooth(m1, ml=ml, sl=sl, mh=mh, sh=sh)+logf_smooth(m2, ml=ml, sl=sl, mh=mh, sh=sh)-2*logf_smooth(30, ml=ml, sl=sl, mh=mh, sh=sh)-2*np.log(30)

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

def logf_smooth(m, ml=5, sl=0.1, mh=45, sh=0.1):
    return np.log(ss.norm().cdf((np.log(m)-np.log(ml))/sl))+np.log((1-ss.norm().cdf((np.log(m)-np.log(mh))/sh)))


def CCfast(alpha=0.75, beta=0, ml=5, sl=0.1, mh=45, sh=0.1):
  
    if (beta!=-1) & (alpha!=1) : 
        return (( mh**(-alpha+beta+2)- ml**(-alpha+beta+2) )/(-alpha+beta+2) - ml**(beta+1) *(mh**(-alpha+1)-ml**(-alpha+1))/(-alpha+1) )/(beta+1)
    else:
        raise ValueError
