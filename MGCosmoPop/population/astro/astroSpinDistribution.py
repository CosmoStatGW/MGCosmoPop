#!/usr/bin/env python3
#    Copyright (c) 2021 Michele Mancarella <michele.mancarella@unige.ch>
#
#    All rights reserved. Use of this source code is governed by a modified BSD
#    license that can be found in the LICENSE file.


from ..ABSpopulation import BBHDistFunction
import numpy as np
from scipy.linalg import inv, det
from scipy.stats import truncnorm, multivariate_normal
from scipy.special import erfc
from scipy.stats import beta
########################################################################
# SPIN DISTRIBUTION
########################################################################



def trunc_gaussian_logpdf(x, mu = 1, sigma = 1, lower = 0, upper=100):

    where_compute= (x>lower) & (x<upper)
    
    pdf=np.empty_like(x)
    pdf[~where_compute]=np.NINF
    x=x[where_compute]
    
    Phialpha = 0.5*erfc(-(lower-mu)/(np.sqrt(2)*sigma))
    Phibeta = 0.5*erfc(-(upper-mu)/(np.sqrt(2)*sigma))
    
    pdf[where_compute] = -np.log(2*np.pi)/2-np.log(sigma)-np.log(Phibeta-Phialpha) -(x-mu)**2/(2*sigma**2)
    
    return pdf



class DummySpinDist(BBHDistFunction):
    
    def __init__(self, ):
        self.nVars=0
        BBHDistFunction.__init__(self)
        
    
    def logpdf(theta, lambdaBBHmass):
        #chi1, chi2 =theta
        return np.zeros(theta[0].shape[0])
    

def get_truncnorm(a, b, mu, sigma):
    
    aa, bb = (a - mu) / sigma, (b - mu) / sigma
    
    return truncnorm( aa, bb, loc=mu, scale=sigma)


    
    
class GaussSpinDist(BBHDistFunction):
    
    def __init__(self, ):
        BBHDistFunction.__init__(self)
        self.params = ['muEff', 'sigmaEff', 'muP', 'sigmaP', 'rho' ] # For the moment we ignore correlation
        
        self.baseValues = {
                           
                           'muEff':0.06,
                           'sigmaEff':0.12, 
                           'muP':0.21, 
                           'sigmaP':0.09, 
                           'rho':0.,
                           }
        
        self.names = {
                           'muEff':r'$\mu_{eff}$',
                           'sigmaEff':r'$\sigma_{eff}$', 
                           'muP':r'$\mu_{p}$', 
                           'sigmaP':r'$\sigma_{p}$',
                           'rho':r'$\rho$'}
         
        self.n_params = len(self.params)
    
        self.maxChiEff = 1
        self.minChiEff = -1
        
        self.maxChiP = 1
        self.minChiP = 0
        self.nVars = 2
        
        print('Gaussian spin distribution base values: %s' %self.baseValues)
    
    
    def logpdf(self, theta, lambdaBBHspin):
        
        chiEff, chiP = theta
        muEff, sigmaEff, muP, sigmaP, rho = lambdaBBHspin
        
        #mean = np.array([muEff, muP])
        #C = np.array( [[sigmaEff**2, rho*sigmaEff*sigmaP ], [rho*sigmaEff*sigmaP , sigmaP**2]]  )
        
        #logpdf = -np.log(2*np.pi)-0.5*np.log(det(C))-0.5*(theta-mean).dot(inv(C)).dot(theta-mean)
        #logpdf = multivariate_normal.logpdf(theta, mean=mean, cov=C )
        
        if rho==0:
            where_compute=~np.isnan(chiP)
            if where_compute.sum==0:
                pdf2=np.full(chiP.shape, 0.)
            else:
            #pdftot=np.empty_like(chiEff)
                pdf2=np.empty_like(chiP)
                pdf2[~where_compute]=0.
                chiP = chiP[where_compute]
                # Put zero (i.e. ignore the effect) when chi_p is not available - this is used when computing selection effects, for which we don't have chi_p
                pdf2[where_compute] =  trunc_gaussian_logpdf(chiP, lower=self.minChiP, upper=self.maxChiP, mu=muP, sigma=sigmaP )
            
            pdf1 = trunc_gaussian_logpdf(chiEff, lower=self.minChiEff, upper=self.maxChiEff, mu=muEff, sigma=sigmaEff ) #get_truncnorm(self.minChiEff, self.maxChiEff, muEff, sigmaEff ).logpdf(chiEff)
            
            
            pdftot = pdf1+pdf2

        else:
 
            where_inf =  ( chiEff < -1 ) | ( chiEff > 1 ) | ( chiP <0 ) | ( chiP > 1 )
            
            sEsP = rho*sigmaEff*sigmaP
            C = np.asarray( [ [sigmaEff**2, sEsP] , [ sEsP, sigmaP**2 ] ] )
            mean = np.asarray( [muEff, muP] )
            x = np.asarray( [chiEff, chiP] ).T
            pdftot =  np.diag(-0.5*(x-mean)@(inv(C))@((x-mean).T)) - np.log(2*np.pi)-0.5*np.log(det(C))
        
            pdftot =  np.where( where_inf, np.NINF, pdftot)
        
        return pdftot

    
    def sample(self, nSamples, lambdaBBHspin, ):

        muEff, sigmaEff, muP, sigmaP, rho = lambdaBBHspin

        sEsP = rho*sigmaEff*sigmaP
        C = np.asarray( [ [sigmaEff**2, sEsP] , [ sEsP, sigmaP**2 ] ] )

        samples_ = multivariate_normal.rvs(mean=[muEff, muP], cov=C, size=nSamples)
        chiesamples, chipsamples = samples_[:, 0], samples_[:, 1]
        to_replace =  (chiesamples < -1)  |  (chiesamples > 1)  | (chipsamples <0)  | (chipsamples > 1) 
        n_replace = to_replace.sum()
        
        while(n_replace>0):
        
            ss = multivariate_normal.rvs(mean=[muEff, muP], cov=C, size=n_replace)
            try:
                chiesamples_, chipsamples_ = ss[:, 0], ss[:, 1]
            except IndexError:
                chiesamples_, chipsamples_ = ss[0], ss[1]
        
            samples_[to_replace, 0] = chiesamples_
            samples_[to_replace, 1] = chipsamples_
        
            chiesamples, chipsamples = samples_[:, 0], samples_[:, 1]
                    
            to_replace =  (chiesamples < -1)  |  (chiesamples > 1)  | (chipsamples < 0)  | (chipsamples > 1) 
            
            n_replace = to_replace.sum()

        return samples_.T
    
  
    
class UniformSpinDistChiz(BBHDistFunction): 
    
    '''
    Uniform distribution in chi1z, chi2z, uncorrelated
    '''
    
    def __init__(self, ):
        BBHDistFunction.__init__(self)
        self.params = ['chiMin', 'chiMax', ] #'rho' ] # For the moment we ignore correlation
        
        self.baseValues = {
                           
                           'chiMin': -0.75,
                           'chiMax':0.75, 
                           }
        
        self.names = {
                           'chiMin':r'$\chi_{ z, Min}$',
                           'chiMax':r'$\chi_{ z, Max}$', }
         
        self.n_params = len(self.params)
    
        self.maxChi = 1
        self.minChi = -1
        self.nVars = 2
        

        
        print('Uniform spin distribution in (chi1z, chi2z) base values: %s' %self.baseValues)
    
    
    def logpdf(self, theta, lambdaBBHspin):
        
        chi1z, chi2z = theta
        chiMin, chiMax, = lambdaBBHspin
        lognorm = -np.log(chiMax-chiMin)
        
        logpdf1 = np.full(chi1z.shape, np.NINF)
        mask1 = (chi1z>chiMin) & (chi1z<chiMax)
        logpdf1[mask1] = lognorm
        
        logpdf2 = np.full(chi2z.shape, np.NINF)
        mask2 = (chi2z>chiMin) & (chi2z<chiMax)
        logpdf2[mask2] = lognorm
        

        return logpdf1+logpdf2
    
    def sample(self, nSamples, lambdaBBHspin, ):
        
        chiMin, chiMax, = lambdaBBHspin
        
        c1 = np.random.uniform(low=chiMin, high=chiMax, size=nSamples)
        c2 = np.random.uniform(low=chiMin, high=chiMax, size=nSamples)
        return c1, c2
    
    
    
class DefaultSpinModel(BBHDistFunction): 
    
    '''
    default spin model, see 2010.14533 app D.1
    '''
    
    def __init__(self, alpha_beta_prior=False):
        BBHDistFunction.__init__(self)
        self.params = ['muChi', 'varChi','zeta','sigmat' ] #'rho' ] # For the moment we ignore correlation
        
        self.baseValues = {
                           
                           'muChi': 0.3,
                           'varChi':0.03, 
                           'zeta': 0.76, 
                           'sigmat':0.8
                           }
        
        self.names = {
                           'muChi':r'$\mu_{ \chi}$',
                           'varChi':r'$\sigma_{ \chi}^2$', 
                           'zeta':r'$\zeta$', 
                           'sigmat':r'$\sigma_t$'
                           }
         
        self.n_params = len(self.params)
    
        self.maxChi = 1
        self.minChi = -1
        self.nVars = 4
        self.alpha_beta_prior=alpha_beta_prior
        

        
        print('Default spin distribution in (chi1, chi2, cost1, cost2) base values: %s' %self.baseValues)
    
    
    def logpdf(self, theta, lambdaBBHspin):
        
        chi1, chi2, cost1, cost2 = theta
        muChi, varChi, zeta, sigmat = lambdaBBHspin

        kappa_ = muChi*(1-muChi)/varChi-1
        
        alphaChi = muChi*kappa_ 
        betaChi = (1-muChi)*kappa_ 

        # This is not part of the distribution, only a restriction of the prior in the LVK analysis
        if self.alpha_beta_prior:
            if (alphaChi<=1) | (betaChi<=1):
                return np.full(chi1.shape, -np.inf)
        
        where_compute= (cost1>=-1) & (cost1<=1) & (cost2>=-1) & (cost2<=1)& (chi1>=0) & (chi1<=1) & (chi2>=0) & (chi2<=1)
    
        logpdf=np.empty_like(cost1)
        logpdf[~where_compute]=np.NINF
        chi1, chi2, cost1, cost2 = chi1[where_compute], chi2[where_compute], cost1[where_compute], cost2[where_compute]
        
        logpdfampl =  beta.logpdf(chi1, alphaChi, betaChi)+beta.logpdf(chi2, alphaChi, betaChi)
        
        pdfcos1 =  np.exp(trunc_gaussian_logpdf(cost1, mu = 1., sigma = sigmat, lower = -1, upper=1)) 
        
        pdfcos2 =  np.exp(trunc_gaussian_logpdf(cost2, mu = 1., sigma = sigmat, lower = -1, upper=1)) 

        logpdftilt = np.log( (1-zeta)/4 + zeta*pdfcos1*pdfcos2 )
        
        logpdf[where_compute] = logpdfampl+logpdftilt
        
        return logpdf
    
    
    def sample(self, nSamples, lambdaBBHspin):
        
        muChi, varChi, zeta, sigmat = lambdaBBHspin

        kappa_ = muChi*(1-muChi)/varChi-1
        
        alphaChi = muChi*kappa_ 
        betaChi = (1-muChi)*kappa_
        
        chi1sam = np.random.beta(alphaChi, betaChi, size=nSamples)
        chi2sam = np.random.beta(alphaChi, betaChi, size=nSamples)
        
        mixture_idx = np.random.choice( 2, size=nSamples, replace=True, p=np.squeeze(np.array([zeta, 1-zeta])) )
        
        myclip_a = -1.
        myclip_b = 1.
        a, b = (myclip_a - 1.) / sigmat, (myclip_b - 1.) / sigmat
        
        cost1sam = np.zeros(nSamples)
        cost2sam = np.zeros(nSamples)
        cost1sam[mixture_idx==0] = truncnorm.rvs(a, b, loc=1., scale=sigmat, size=nSamples-mixture_idx.sum(), )
        cost1sam[mixture_idx==1] = np.random.uniform(low=-1., high=1.0, size=mixture_idx.sum())
        
        cost2sam[mixture_idx==0] = truncnorm.rvs(a, b, loc=1., scale=sigmat, size=nSamples-mixture_idx.sum(), )
        cost2sam[mixture_idx==1] = np.random.uniform(low=-1., high=1.0, size=mixture_idx.sum())
        
        
        return chi1sam, chi2sam, cost1sam, cost2sam



class UniformOnSphereSpin(BBHDistFunction): 
    
    '''
    Uniform distribution on the sphere
    '''
    
    def __init__(self, ):
        BBHDistFunction.__init__(self)
        self.params = [ 'chiMax', ] 
        
        self.baseValues = {                            
                           'chiMax':0.998, 
                           }
        
        self.names = { 'chiMax':r'$\chi_{ z, Max}$', }
         
        self.n_params = len(self.params)
    
        self.maxChi = 1
        self.minChi = -1
        self.nVars = 6
        
        
        print('Uniform spin distribution on the sphere. Base values: %s' %self.baseValues)
    
    
    def logpdf(self, theta, lambdaBBHspin):
        
        s1x, s1y, s1z, s2x, s2y, s2z = theta
        chiMax = lambdaBBHspin[0]
        
        s12 = s1x**2 + s1y**2 + s1z**2 
        s22 = s2x**2 + s2y**2 + s2z**2
        
        logpfds1 =  np.where(s12 < chiMax**2, - np.log(4 * np.pi) - np.log(chiMax) - np.log(s12), np.NINF)
        logpfds2 =  np.where(s22 < chiMax**2, - np.log(4 * np.pi) - np.log(chiMax) - np.log(s22), np.NINF)

        return logpfds1+logpfds2

    
    def sample(self, nSamples, lambdaBBHspin, ):
        
        chiMax = lambdaBBHspin
        
        costheta1 = np.random.uniform(-1, 1, nSamples)            
        theta1 = np.arccos(costheta1)

        costheta2 = np.random.uniform(-1, 1, nSamples)            
        theta2 = np.arccos(costheta2)

        phi1 = np.random.uniform(0, np.pi*2, nSamples)
        phi2 = np.random.uniform(0, np.pi*2, nSamples)

        s1 = np.random.uniform(0, chiMax, nSamples)
        s2 = np.random.uniform(0, chiMax, nSamples)

        s1x = s1*np.sin(theta1)*np.cos(phi1)
        s1y = s1*np.sin(theta1)*np.sin(phi1)
        s1z = s1*np.cos(theta1)

        s2x = s2*np.sin(theta2)*np.cos(phi2)
        s2y = s2*np.sin(theta2)*np.sin(phi2)
        s2z = s2*np.cos(theta2)
        
        return s1x, s1y, s1z, s2x, s2y, s2z

