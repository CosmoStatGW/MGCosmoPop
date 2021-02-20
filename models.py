"""
Created on Wed Jan 20 16:38:28 2021

@author: Michi
"""

#from Globals import *
import Globals
import config
#from config import *
import utils
#from  cosmo import *
import cosmo
import data
import scipy.stats as ss
from getLambda import get_Lambda
from astropy.cosmology import  Planck15, z_at_value
#from scipy.integrate import cumtrapz
#import astropy.units as u
import numpy as np
from scipy.special import logsumexp
#####################################################
#####################################################

print('Loading data...')
theta, Nsamples = data.load_data(config.dataset_name)
logNsamples = np.log(Nsamples)
Nobs = theta[0].shape[0]
logNobs = np.log(Nobs)
m1z, m2z, dL = theta
assert (m1z > 0).all()
assert (m2z > 0).all()
assert (dL > 0).all()

assert(m2z<m1z).all()


print('theta shape: %s' % str(theta.shape))
print('We have %s observations' % Nobs)

print('Number of samples: %s' %Nsamples )


print('Loading injections from %s dataset...' %config.dataset_name_injections)
theta_sel, weights_sel, N_gen, Tobs, ifars = data.load_injections_data(config.dataset_name_injections)
log_weights_sel = np.log(weights_sel)
logTobs=np.log(Tobs)
if config.dataset_name_injections=='mock':
    m1z_sel, m2z_sel, dL_sel = theta_sel
else:
    #m1_sel, m2_sel, z_sel = theta_sel
    m1z_sel, m2z_sel, dL_sel = theta_sel
    gstlal_ifar, pycbc_ifar, pycbc_bbh_ifar = ifars
logN_gen=np.log(N_gen)
print('Number of total injections: %s' %N_gen)
print('Number of detected injections: %s' %weights_sel.shape[0])
#if config.dataset_name_injections=='mock':
zmax=z_at_value(Planck15.luminosity_distance, dL_sel.max()*Globals.which_unit)
#else:
#zmax=z_sel.max()
print('Max z of injections: %s' %zmax)
print('Oberving time: %s years' %Tobs)



#OrMassPrior =  data.originalMassPrior(m1z, m2z)
#OrDistPrior  = data.originalDistPrior(dL)
logOrMassPrior =  data.logOriginalMassPrior(m1z, m2z)
logOrDistPrior  = data.logOriginalDistPrior(dL)



#####################################################
#####################################################



#####################################################




def logSelectionBias(Lambda, m1, m2, z, get_neff=False, verbose=False,):
    
    #m1, m2, z = precomputed['m1'], precomputed['m2'], precomputed['z']
    #Lambda = get_Lambda(Lambda_test, Lambda_ntest)
    #H0, Om0, w0, Xi0, n, R0, lambdaRedshift, alpha, beta, ml, sl, mh, sh = Lambda
    if config.dataset_name_injections=='mock':
        logxx = log_dN_dm1zdm2zddL(Lambda, m1, m2, z)    
    else:
        condition =  (gstlal_ifar > config.ifar_th) | (pycbc_ifar > config.ifar_th) | (pycbc_bbh_ifar > config.ifar_th)
        # LVC gives samples and weights in source frame quantities (m1, m2, z), so no Jacobian needed 
        logxx = np.where( condition, log_dN_dm1zdm2zddL(Lambda, m1, m2, z)+np.log(0.25) ,  np.NINF) # add 1/2 for each spin dimension
    
    logxx -= log_weights_sel
    logMu = np.logaddexp.reduce(logxx) - logN_gen
    
    if np.isnan(logMu):
        raise ValueError('NaN value for logMu. Values of Lambda: %s%s' %(str(config.allMyPriors.allParams), str(Lambda) ) )
    if not get_neff:
        return logMu, np.NaN #np.repeat(np.NaN, logMu.shape[0] )
    else:
        #muSq = np.exp(2*logMu)#.astype('float128')
        logs2 = ( np.logaddexp.reduce(2*logxx) -2*logN_gen)#.astype('float128')
        
        #np.logaddexp.reduce(2.0*log_dN - 2.0*log(p_draw)) - 2.0*log(Ndraw)
        #log_sigma2 = logdiffexp(log_s2, 2.0*log_mu - log(Ndraw))
        #Neff = exp(2.0*log_mu - log_sigma2)
        
        logSigmaSq = utils.logdiffexp( logs2, 2.0*logMu - logN_gen )#.astype('float128') #-logN_gen-2*logMu+logsumexp(2*xx)
        #SigmaSq = np.exp(logs2) - muSq/N_gen
        #Neff = muSq/SigmaSq
        Neff = np.exp( 2.0*logMu - logSigmaSq)
        #if np.isnan(Neff):
        #    raise ValueError('NaN value for logMu. Values of Lambda: %s%s' %(str(config.allMyPriors.allParams), str(Lambda) ) )
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
    #print(m1, m2, z)
    #print('logLik_ shape: %s' %str(logLik_.shape))
    #return np.log(lik.mean(axis=1)) .sum(axis=(-1))
    allLogLiks = np.logaddexp.reduce(logLik_, axis=-1)-logNsamples
    #allLogLiks = logsumexp(logLik_, axis=-1)-logNsamples
    #print('allLogLiks shape: %s' %str(allLogLiks.shape))
    ll=allLogLiks.sum()
    #print('ll1 logsumexp: %s' %ll)
    #print('ll2 np: %s' %(np.logaddexp.reduce(logLik_, axis=-1)-logNsamples).sum())
    #lik = np.exp(logLik_) #dN_dm1zdm2zddL(Lambda, m1, m2, z)
    #print('log ll1 : %s' % np.log( lik.mean(axis=1)).sum(axis=(-1)))
    
    if np.isnan(ll):
        raise ValueError('NaN value for logLik. Values of Lambda: %s%s' %(str(config.allMyPriors.allParams), str(Lambda) ) )

    return ll #allLogLiks.sum()


def log_prior(Lambda_test, priorLimits, params_inference, pNames, pParams):
    
    #  = [ (priorLimits.limInf[param], priorLimits.limSup[param] ) for param in priorLimits.names ]
    
    if np.isscalar(Lambda_test):
        #limInf = priorLimits.limInf[params_inference]
        #limSup = priorLimits.limSup[params_inference]#priorLimits[0]
        limInf, limSup =  priorLimits[0] #config.myPriorLimits[0]
        condition = limInf < Lambda_test < limSup
    else:
        condition = True
        for i,(limInf, limSup) in enumerate(priorLimits): #(config.myPriorLimits): #param in enumerate(params_inference): #(limInf, limSup) in enumerate(priorLimits):
            #limInf = priorLimits.limInf[param]
            #limSup = priorLimits.limSup[param]
            condition &= limInf < Lambda_test[i] < limSup
    
    if not condition:
        return np.NINF
    
    
    lp = 0
    for i,param in enumerate(params_inference):
            pname= pNames[param]
            if np.isscalar(Lambda_test):
                    x = Lambda_test
            else:
                    x=Lambda_test[i]
            if pname=='flatLog':
                lp-=np.log(x)
            elif pname=='gauss':
                #x = Lambda_test[i]
                mu, sigma = pParams[param]['mu'], pParams[param]['sigma']
                if np.abs(x-mu)>7*sigma:
                    return np.NINF
                lp+= (-np.log(sigma)-(x-mu)**2/(2*sigma**2)) 
    return lp#func(Lambda_test, params_inference) #config.allMyPriors.get_logVals(Lambda_test, params_inference) # sum of log priors of all the variables





def log_posterior(Lambda_test, Lambda_ntest,  priorLimits, params_inference, pNames, pParams, return_all):
    
    
    lp = log_prior(Lambda_test,  priorLimits, params_inference, pNames, pParams)
    if not np.isfinite(lp):
        return -np.inf
    
    Lambda = get_Lambda(Lambda_test, Lambda_ntest)
    # Compute source frame masses and redshifts
    m1_obs, m2_obs, z_obs = get_mass_redshift(Lambda, which_data='obs')
    ll = logLik(Lambda, m1_obs, m2_obs, z_obs )
    
    logPost= ll+lp
    
    ### Selection bias
    #if config.with_jacobian_inj:
    m1_inj, m2_inj, z_inj = get_mass_redshift(Lambda, which_data='inj')
    #else:
    #    m1_inj, m2_inj, z_inj  = m1_sel, m2_sel, z_sel

    logMu, Neff = logSelectionBias(Lambda, m1_inj, m2_inj, z_inj, get_neff = config.selection_integral_uncertainty, verbose=config.verbose_bias )
    
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
        mu = np.exp(logMu)
        #R0 = np.exp(logR0)
        logPost -= R0*mu
        if config.selection_integral_uncertainty:
            logPost+= (R0*mu)*(R0*mu)/ (2 * Neff)#-ss.norm(loc=mu, scale=mu/np.sqrt(Neff) ).logsf(0)+ss.norm(loc=R0*mu**2/Neff-mu, scale=mu/np.sqrt(Neff)).logsf(0)
            
    
    if np.isnan(logPost):
        raise ValueError('NaN value for logPost. Values of Lambda: %s%s' %(str(config.allMyPriors.allParams), str(Lambda) ) )
    if not return_all:
        return logPost
    return logPost, ll, lp, logMu, Neff




#####################################################

def get_mass_redshift(Lambda, which_data):
    '''
    Compute only once some quantities that go into selection effects and likelihood
    '''
    #if Lambda is not None:
    H0, Om0, w0, Xi0, n, R0, lambdaRedshift, alpha, beta, ml, sl, mh, sh = Lambda
    
    if which_data=='obs':      
        z = get_redshift(  dL, H0, Om0, w0, Xi0, n)
        m1 = m1z / (1 + z)    
        m2 = m2z / (1 + z)
    elif which_data=='inj':
        #if config.with_jacobian_inj:
        #print('Precomuting with inj data')
        z = get_redshift( dL_sel, H0, Om0, w0, Xi0, n)
        m1 = m1z_sel / (1 + z)    
        m2 = m2z_sel / (1 + z)
        #else:
        #    raise ValueError('No need to compute redshift and mass')
        
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




def log_dN_dm1dm2dz(Lambda, m1, m2, z):
    """
    - theta is an array (m1z, m2z, dL) where m1z, m2z, dL are arrays 
    of the GW posterior samples
     
     Lambda = (H0, Xi0, n, lambdaRedshift, lambdaBBH ) 
     lambdaBBH is the parameters of the BBH mass function 
    """
    H0, Om0, w0, Xi0, n, R0, lambdaRedshift, alpha, beta, ml, sl, mh, sh = Lambda
    lambdaBBH = [alpha, beta, ml, sl, mh, sh]
    #mmin = ml-15*sl
    #mmax = mh+15*sh
    #where_compute = (mmin < m2) & (m1 < mmax) #& (m2 < m1) 
    #return np.where(where_compute, logTobs+cosmo.log_dV_dz(z, H0, Om0, w0)+log_dtobsdtdet(z) +log_dNdVdt(z, lambdaRedshift)+logMassPrior(m1, m2, lambdaBBH), np.NINF)
    return logTobs+cosmo.log_dV_dz(z, H0, Om0, w0)+log_dtobsdtdet(z) +log_dNdVdt(z, lambdaRedshift)+logMassPrior(m1, m2, lambdaBBH)

def log_dtobsdtdet(z):
    return -np.log1p(z)


def log_dNdVdt(z, lambdaRedshift):
    '''
    Evolution of total rate with redshift: (1+z)**lambda
    The overall normalization is added in log_posterior, in order to allow for analytic marginalisation on it
    '''
    return lambdaRedshift*np.log1p(z)
    
    

def log_dN_dm1zdm2zddL(Lambda, m1, m2, z):
    H0, Om0, w0, Xi0, n, R0, lambdaRedshift, alpha, beta, ml, sl, mh, sh = Lambda
    #return dN_dm1dm2dz(Lambda, m1, m2) / ( MsourceToMdetJacobian(z) * ddL_dz(z, H0, Om0, w0, Xi0, n) )
    #lambdaBBH = [alpha, beta, ml, sl, mh, sh]
    #mmin = ml-15*sl
    #mmax = mh+15*sh
    #where_compute =  (mmin < m2) & (m1 < mmax)  # (m2 < m1)
    #return np.where(where_compute, log_dN_dm1dm2dz(Lambda, m1, m2, z)-log_dMsourcedMdet(z) - cosmo.log_ddL_dz(z, H0, Om0, w0, Xi0, n), np.NINF)  
    return log_dN_dm1dm2dz(Lambda, m1, m2, z)-log_dMsourcedMdet(z) - cosmo.log_ddL_dz(z, H0, Om0, w0, Xi0, n)

def log_dMsourcedMdet(z):
    return 2*np.log1p(z)





#####################################################


def logMassPrior(m1, m2, lambdaBBH):
    alpha, beta, ml, sl, mh, sh = lambdaBBH
    if config.mass_normalization=='integral' and config.mass_model=='Farr':
        return np.log(m1)*(-alpha)+np.log(m2)*(beta)+logf_smooth(m1, ml=ml, sl=sl, mh=mh, sh=sh)+logf_smooth(m2, ml=ml, sl=sl, mh=mh, sh=sh)-np.log(CCfast(alpha, beta, ml, sl, mh, sh))
    elif config.mass_normalization=='pivot' and config.mass_model=='Farr':
        return np.log(m1)*(-alpha)+alpha*np.log(30) +np.log(m2)*(beta) -beta*np.log(30) +logf_smooth(m1, ml=ml, sl=sl, mh=mh, sh=sh)+logf_smooth(m2, ml=ml, sl=sl, mh=mh, sh=sh)-2*logf_smooth(30, ml=ml, sl=sl, mh=mh, sh=sh)-2*np.log(30)
    elif config.mass_model=='LVC':
        mmin = ml#-7*sl
        mmax = mh#+7*sh
        where_compute = (m2 < m1) & (mmin < m2) & (m1 < mmax)
        return  np.where( where_compute,  (-alpha)*np.log(m1)  + beta*np.log(m2) -logC1(m1, beta, ml) -logC2(alpha, ml, mh),  np.NINF) # -beta*np.log(m1)
    else:
        raise ValueError


def logC1(m, beta, ml):
    if beta>-1:
        return -np.log1p(beta)+utils.logdiffexp((1+beta)*np.log(m), (1+beta)*np.log(ml)) # -beta*np.log(m)
    elif beta<-1:
        return -np.log(-1-beta)+utils.logdiffexp( (1+beta)*np.log(ml), (1+beta)*np.log(m)) #-beta*np.log(m)
    raise ValueError # 1 / m / np.log(m / ml)


def logC2(alpha, ml, mh):
    if (alpha < 1) & (alpha!=0):
        return -np.log1p(-alpha)+utils.logdiffexp( (1-alpha)*np.log(mh), (1-alpha)*np.log(ml) ) #(1 - alpha) / (mh ** (1 - alpha) - ml ** (1 - alpha))
    #return 1 / np.log(mh / ml)
    elif (alpha > 1) :
        return -np.log(alpha-1)+utils.logdiffexp(  (1-alpha)*np.log(ml), (1-alpha)*np.log(mh) )
    raise ValueError

def logf_smooth(m, ml=5, sl=0.1, mh=45, sh=0.1):
    return np.log(ss.norm().cdf((np.log(m)-np.log(ml))/sl))+np.log((1-ss.norm().cdf((np.log(m)-np.log(mh))/sh)))


def CCfast(alpha=0.75, beta=0, ml=5, sl=0.1, mh=45, sh=0.1):
  
    if (beta!=-1) & (alpha!=1) : 
        return (( mh**(-alpha+beta+2)- ml**(-alpha+beta+2) )/(-alpha+beta+2) - ml**(beta+1) *(mh**(-alpha+1)-ml**(-alpha+1))/(-alpha+1) )/(beta+1)
    else:
        raise ValueError
        
        
        
#####################################################
# Not used
#####################################################


def f_smooth(m, ml=5, sl=0.1, mh=45, sh=0.1):
    return ss.norm().cdf((np.log(m) - np.log(ml)) / sl) * (1 - ss.norm().cdf((np.log(m) - np.log(mh)) / sh))


def C1(m, beta, ml):
    if beta != -1:
        return (1 + beta) / m / (1 - (ml / m) ** (1 + beta))
    return 1 / m / np.log(m / ml)


def C2(alpha, ml, mh):
    if alpha != 1:
        return (1 - alpha) / (mh ** (1 - alpha) - ml ** (1 - alpha))
    return 1 / np.log(mh / ml)


def massPrior(m1, m2, lambdaBBH):
    """
    lambdaBBH is the array of parameters of the BBH mass function 
    """
    alpha, beta, ml, sl, mh, sh = lambdaBBH
    if config.mass_normalization=='integral':
        return (m1)**(-alpha)*(m2)**(beta)*f_smooth(m1, ml=ml, sl=sl, mh=mh, sh=sh)*f_smooth(m2, ml=ml, sl=sl, mh=mh, sh=sh)/CCfast(alpha, beta, ml, sl, mh, sh)
        #return m1 ** (-alpha) * (m2 / m1) ** beta * f_smooth(m1, ml=ml, sl=sl, mh=mh, sh=sh) * f_smooth(m2, ml=ml, sl=sl, mh=mh, sh=sh) * C1(m1, beta, ml) * C2(alpha, ml, mh)
    elif config.mass_normalization=='pivot':
        return (m1/30)**(-alpha)*(m2/30)**(beta)*f_smooth(m1, ml=ml, sl=sl, mh=mh, sh=sh)*f_smooth(m2, ml=ml, sl=sl, mh=mh, sh=sh) /f_smooth(30, ml=ml, sl=sl, mh=mh, sh=sh)**2/(30**2)
    else:
        raise ValueError



def rateDensityEvol(z, lambdaRedshift):
    """
    merger rate density evolution in redshift (un-normalized)
    """
    return (1 + z)**(lambdaRedshift)



def MsourceToMdetJacobian(z):
    return (1 + z)*(1 + z)


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
    #lambdaBBH = [alpha, beta, ml, sl, mh, sh]
    return dN_dm1dm2dz(Lambda, m1, m2, z) /  MsourceToMdetJacobian(z) / cosmo.ddL_dz(z, H0, Om0, w0, Xi0, n) 




def selectionBias(Lambda, m1, m2, z, get_neff=False, verbose=False, ):
    
    #m1, m2, z = precomputed['m1'], precomputed['m2'], precomputed['z']
    #Lambda = get_Lambda(Lambda_test, Lambda_ntest)
    #H0, Om0, w0, Xi0, n, R0, lambdaRedshift, alpha, beta, ml, sl, mh, sh = Lambda
    
    #if config.dataset_name_injections=='mock':
    xx = dN_dm1zdm2zddL(Lambda, m1, m2, z)/weights_sel
    #else:
    #    xx = dN_dm1dm2dz(Lambda, m1, m2, z)/weights_sel
        
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
