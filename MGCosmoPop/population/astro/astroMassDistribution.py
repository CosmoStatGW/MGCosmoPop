#!/usr/bin/env python3
#    Copyright (c) 2021 Michele Mancarella <michele.mancarella@unige.ch>
#
#    All rights reserved. Use of this source code is governed by a modified BSD
#    license that can be found in the LICENSE file.

from ..ABSpopulation import BBHDistFunction
import numpy as np
import scipy.stats as ss
from scipy.integrate import cumtrapz

import sys
import os

PACKAGE_PARENT = '../..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import utils



########################################################################
# MASS DISTRIBUTION
########################################################################


def truncated_power_law(m, alpha, ml, mh):
        where_nan = np.isnan(m)
        result = np.zeros(m.shape)
        

        result[where_nan]=np.NINF
        
        where_compute = (ml < m) & (m < mh )
        result[~where_compute] = np.NINF
        
        m = m[where_compute]
        result[where_compute] = np.log(m)*(-alpha)
        
        return result


def norm_truncated_pl(alpha, ml, mh):
            
        if (alpha < 1) & (alpha!=0):
            return -np.log1p(-alpha)+utils.logdiffexp( (1-alpha)*np.log(mh), (1-alpha)*np.log(ml) ) #(1 - alpha) / (mh ** (1 - alpha) - ml ** (1 - alpha))

        elif (alpha > 1) :
            return -np.log(alpha-1)+utils.logdiffexp(  (1-alpha)*np.log(ml), (1-alpha)*np.log(mh) )
        raise ValueError
        


class AstroSmoothPowerLawMass(BBHDistFunction):
    
    '''
    Mass distribution  - Power law with smooth transition to zero at the edges; from 1908.09084
    '''
    
    
    def __init__(self, normalization='integral' ):
        
        BBHDistFunction.__init__(self)
        
        self.normalization=normalization
        print('Normalization of the mass function: %s' %normalization)
        self.params = ['alpha', 'beta', 'ml', 'sl', 'mh', 'sh' ]
        
        self.baseValues = {
                           
                           'alpha':0.75,
                           'beta':0.0, 
                           'ml':5.0, 
                           'sl':0.1, 
                           'mh':45.0,
                           'sh':0.1}
        
        self.names = {
                           'alpha':r'$\alpha$',
                           'beta':r'$\beta$', 
                           'ml':r'$M_l$', 
                           'sl':r'$\sigma_l$', 
                           'mh':r'$M_h$',
                           'sh':r'$\sigma_h$'}
         
        self.n_params = len(self.params)
    
    def _logpdfm1only(self, m1, alpha, ml, sl, mh, sh ):
        
        logp = np.log(m1)*(-alpha)+self._logf_smooth(m1, ml=ml, sl=sl, mh=mh, sh=sh)
        
        return logp
    
    
    
    def _logpdfm1(self, m1, alpha, ml, sl, mh, sh ):
        '''Marginal prob. '''
        
        logp = np.log(m1)*(-alpha)+self._logf_smooth(m1, ml=ml, sl=sl, mh=mh, sh=sh)
        
        return logp
    
    def _logpdfm2(self, m2, beta, ml, sl, mh, sh ):
        
        logp = np.log(m2)*(beta)+self._logf_smooth(m2, ml=ml, sl=sl, mh=mh, sh=sh)
        
        return logp
    
    
    def logpdf(self, theta, lambdaBBHmass):
        
        '''p(m1, m2 | Lambda ), normalized to one'''
        
        m1, m2 = theta
        alpha, beta, ml, sl, mh, sh = lambdaBBHmass
        
        logpdfMass = self._logpdfm1only(m1,alpha, ml, sl, mh, sh ) + self._logpdfm2(m2, beta, ml, sl, mh, sh )
        
        if self.normalization=='integral':
            logNorm = -np.log(self._get_normalization(lambdaBBHmass))
        elif self.normalization=='pivot':
            logNorm = alpha*np.log(30)-beta*np.log(30)-2*self._logf_smooth(30, ml=ml, sl=sl, mh=mh, sh=sh)-2*np.log(30)
        
        return logpdfMass+logNorm
        
       
    def _logf_smooth(self, m, ml=5, sl=0.1, mh=45, sh=0.1):
                
        return np.log(ss.norm().cdf((np.log(m)-np.log(ml))/sl))+np.log((1-ss.norm().cdf((np.log(m)-np.log(mh))/sh)))
       
    
    def _get_normalization(self, lambdaBBHmass):
        '''Normalization of p(m1, m2 | Lambda ). Warning: works well only when sl and sh are small '''
        alpha, beta, ml, sl, mh, sh = lambdaBBHmass
        
        if (beta!=-1) & (alpha!=1) : 
            return (( mh**(-alpha+beta+2)- ml**(-alpha+beta+2) )/(-alpha+beta+2) - ml**(beta+1) *(mh**(-alpha+1)-ml**(-alpha+1))/(-alpha+1) )/(beta+1)
        else:
            raise ValueError 
    
    
    def sample(self, nSamples, lambdaBBHmass):
        alpha, beta, ml, sl, mh, sh = lambdaBBHmass
        mMin = 3
        mMax = 100
        
        pm1 = lambda x: np.exp(self._logpdfm1(x, alpha, ml, sl, mh, sh ))
        pm2 = lambda x: np.exp(self._logpdfm2(x, beta, ml, sl, mh, sh ))
        
        m1 = self._sample_pdf(nSamples, pm1, mMin, mMax)
        m2 = self._sample_vector_upper(pm2, mMin, m1)
        
            
        return m1, m2

    


##############################################################################
# Truncated power law


class TruncPowerLawMass(BBHDistFunction):
    
    '''
    Mass distribution  - Truncated Power Law
    '''

    def __init__(self):
        
        BBHDistFunction.__init__(self)
        
        self.params = ['alpha', 'beta', 'ml',  'mh', ]
        
        self.baseValues = {
                           
                           'alpha':0.75,
                           'beta':0.0, 
                           'ml':4.0, 
                           'mh':78.5,
                           }
        
        self.names = {
                           'alpha':r'$\alpha$',
                           'beta':r'$\beta$', 
                           'ml':r'$M_l$', 
                           'mh':r'$M_h$',}
         
        self.n_params = len(self.params)
    
    
    def _logpdfm1(self, m, alpha, ml, mh):
        '''
        Marginal distribution p(m1)
        '''
        return truncated_power_law(m, alpha, ml, mh)
        
        
    
    
    def _logpdfm2(self, m, beta, ml):
        '''
        Conditional distribution p(m2 | m1)
        '''
        where_nan = np.isnan(m)
        result = np.zeros(m.shape)
        

        result[where_nan]=np.NINF
        
        where_compute = (ml < m)
        result[~where_compute] = np.NINF
        
        m = m[where_compute]
        result[where_compute] = np.log(m)*(beta) 
        
        return result
    
    
    def logpdf(self, theta, lambdaBBHmass):
        
        '''p(m1, m2 | Lambda ), normalized to one'''
        
        m1, m2 = theta
        alpha, beta, ml, mh = lambdaBBHmass
        
        where_compute = (m2 < m1) & (ml < m2) & (m1 < mh )
     
        return np.where( where_compute,   self._logpdfm1(m1, alpha, ml, mh) + self._logpdfm2(m2, beta, ml) +self._logC(m1, beta, ml) -  self._logNorm( alpha, ml, mh) ,  np.NINF)
        
    
    
    def _logC(self, m, beta, ml):
        '''
        Gives inverse log integral of  p(m1, m2) dm2 (i.e. log C(m1) in the LVC notation )
        Used if we want to impose that the  distribution _logpdfm1 corresponds to the marginal distribution p(m1)
        '''
        if beta>-1:
            return np.log1p(beta)-utils.logdiffexp((1+beta)*np.log(m), (1+beta)*np.log(ml)) # -beta*np.log(m)
        elif beta<-1:
            return +np.log(-1-beta)-utils.logdiffexp( (1+beta)*np.log(ml), (1+beta)*np.log(m)) #-beta*np.log(m)
        raise ValueError # 1 / m / np.log(m / ml)


    def _logNorm(self, alpha, ml, mh):
        '''
        Gives log integral of  p(m1, m2) dm1 dm2 (i.e. total normalization of mass function )

        '''
        return norm_truncated_pl(alpha, ml, mh)
        

       
    
    def sample(self, nSamples, lambdaBBHmass):
        alpha, beta, ml, mh = lambdaBBHmass

        
        pm1 = lambda x: np.exp(self._logpdfm1(x, alpha, ml, mh ))
        pm2 = lambda x: np.exp(self._logpdfm2(x, beta, ml ))
        
        m1 = self._sample_pdf(nSamples, pm1, ml, mh)
        #m2 = self._sample_pdf(nSamples, pm2, mMin, mMax)
        m2 = self._sample_vector_upper(pm2, ml, m1)
        assert(m2<=m1).all() 
        assert(m2>=ml).all() 
        assert(m1<=mh).all() 
            
        return m1, m2



##############################################################################
# Broken power law


class BrokenPowerLawMass(BBHDistFunction):
    
    '''
    Mass distribution  - Broken Power Law
    '''

    def __init__(self):
        
        BBHDistFunction.__init__(self)
        
        self.params = ['alpha1', 'alpha2', 'beta', 'deltam', 'ml',  'mh', 'b' ]
        
        self.baseValues = {
                           'alpha1':1.6,
                           'alpha2': 5.6 , 
                           'beta': 1.4,
                           'deltam':4.8,
                           'ml':4,
                           'mh':87,
                           'b': 0.43,
                           }
        
        self.names = { 'alpha1': r'$\alpha_1$',
                           'alpha2':  r'$\alpha_2$' , 
                           'beta':  r'$\beta_q$',
                           'deltam': r'$\delta_{\rm m}$',
                           'ml': r'$m_{\rm min}$',
                           'mh':r'$m_{\rm max}$',
                           'b': r'$b$',
                           
                           }
         
        self.n_params = len(self.params)
        
        print('Broken power law mass function base values: %s' %self.baseValues)
    
    
    def _get_Mbreak(self, mMin, mMax, b):
        return  mMin + b*(mMax - mMin)
    
    
    def _logS(self, m, deltam, ml,):
        maskL = m <= ml #- eps
        maskU = m >= (ml + deltam) #+ eps
        s = np.empty_like(m)
        s[maskL] = np.NINF
        s[maskU] = 0
        maskM = ~(maskL | maskU)
        s[maskM] = -np.logaddexp( 0, (deltam/(m[maskM]-ml) + deltam/(m[maskM]-ml - deltam) ) ) #1/(np.exp(deltam/(m[maskM]-ml) + deltam/(m[maskM]-ml - deltam))+1)
        return s
    
    
    def _logpdfm1(self, m,  alpha1, alpha2, deltam, ml, mh, b):
        '''
        Marginal distribution p(m1), not normalised
        '''

        mBreak = self._get_Mbreak( ml, mh, b)
        
        where_nan = np.isnan(m)
        result = np.empty_like(m)
        
        result[where_nan]=np.NINF
        
        where_compute = (m <= mh) & (m >= ml) & (~where_nan)
        result[~where_compute] = np.NINF
        
        m = m[where_compute]
        result[where_compute] = np.where(m < mBreak, np.log(m)*(-alpha1)+self._logS(m, deltam, ml), np.log(mBreak)*(-alpha1+alpha2)+np.log(m)*(-alpha2)+self._logS(m, deltam, ml) )
        
        return result
        
    
    
    def _logpdfm2(self, m2, beta, deltam, ml):
        '''
        Conditional distribution p(m2 | m1)
        '''
        where_nan = np.isnan(m2)
        result = np.empty_like(m2)

        result[where_nan]=np.NINF
        
        where_compute = (ml<= m2) & (~where_nan)
        result[~where_compute] = np.NINF
        
        m2 = m2[where_compute]
        result[where_compute] = np.log(m2)*(beta)+self._logS(m2, deltam, ml)
        return result
        
    
    
    def logpdf(self, theta, lambdaBBHmass, **kwargs):
        
        '''p(m1, m2 | Lambda ), normalized to one'''
        
        m1, m2 = theta
        alpha1, alpha2, beta, deltam, ml, mh, b = lambdaBBHmass
        
        
        where_nan = np.isnan(m1)
        assert (where_nan==np.isnan(m2)).all()
        result = np.empty_like(m1)

        result[where_nan]=np.NINF
        
        where_compute = (m2 < m1) & (ml< m2) & (m1 < mh ) & (~where_nan)
        result[~where_compute] = np.NINF
        
        m1 = m1[where_compute]
        m2 = m2[where_compute]
        
        
        result[where_compute] = self._logpdfm1(m1,  alpha1, alpha2, deltam, ml, mh, b ) + self._logpdfm2(m2, beta, deltam, ml) + self._logC(m1, beta, deltam,  ml, **kwargs)-  self._logNorm( alpha1, alpha2, deltam, ml, mh, b,)
        return result
        
        
        
    
    
    def _logC(self, m, beta, deltam, ml, res = 200, exact_th=0.):
        '''
        Gives inverse log integral of  p(m1, m2) dm2 (i.e. log C(m1) in the LVC notation )
        '''
        xlow=np.linspace(ml, ml+deltam+deltam/10, 200)
        xup=np.linspace(ml+deltam+deltam/10+deltam/100, m[~np.isnan(m)].max(), res)
        xx=np.sort(np.concatenate([xlow,xup], ))
  
        p2 = np.exp(self._logpdfm2( xx , beta, deltam, ml))
        cdf = cumtrapz(p2, xx)
        
        where_compute = ~np.isnan(m)
        where_exact = m <exact_th*ml
        
        where_approx = (~where_exact) & (where_compute)
        
        
        result = np.empty_like(m)
        
        result[~where_compute]=np.NINF
        result[where_exact]=self._logCexact(m[where_exact], beta, deltam, ml,) #np.NINF
        
        result[where_approx] = -np.log( np.interp(m[where_approx], xx[1:], cdf) )
        
        return result
    
    
    def _logCexact(self, m, beta, deltam, ml, res=1000):
        result=[]

        for mup in m:

            xx = np.linspace(ml, mup, res)
            p2 = np.exp(self._logpdfm2( xx , beta, deltam, ml))
            result.append(-np.log(np.trapz(p2,xx)) )

    
        return np.array(result)
        


    def _logNorm(self, alpha1, alpha2, deltam, ml, mh, b , res=200):
        '''
        Gives log integral of  p(m1, m2) dm1 dm2 (i.e. total normalization of mass function )

        '''
        
        mbr = self._get_Mbreak( ml, mh, b)
        

        ms1 = np.linspace(1., ml+deltam+deltam/10, 200)
        ms2 = np.linspace( ml+deltam+deltam/10+1e-01, mbr-mbr/10, int(res/2) )
        ms3= np.linspace( mbr-mbr/10+1e-01, mbr+mbr/10, 50 )
        ms4 = np.linspace(mbr+mbr/10+1e-01, mh+mh/10, int(res/2) )
        
        ms=np.sort(np.concatenate([ms1,ms2, ms3, ms4], ))
        
        p1 = np.exp(self._logpdfm1( ms ,alpha1, alpha2, deltam, ml, mh, b ))
        return np.log(np.trapz(p1,ms))
        
    
    
    def _logNorm1(self, alpha1, alpha2, deltam, ml, mh, b ):
        '''
        Gives log integral of  p(m1, m2) dm1 dm2 (i.e. total normalization of mass function )

        '''
        mbr = self._get_Mbreak( ml, mh, b)
        x3 = +np.log( mbr**(1-alpha1)*(alpha2-alpha1)-ml**(1-alpha1)*(1-alpha2) +mh**(1-alpha2)*mbr**(alpha2-alpha1)*(1-alpha1) )
        if np.isnan(x3):
            raise ValueError
        
        if (alpha1 < 1) & (alpha2<1) & (alpha1 != 0) & (alpha2!= 0) :
            x1 = -np.log1p(-alpha1)
            x2 = -np.log1p(-alpha2)

        elif (alpha1 > 1) & (alpha2<1) :
            x1 = -np.log(alpha1-1)
            x2 = -np.log1p(-alpha2)

        elif  (alpha1 < 1) & (alpha2>1) :   
            x1 = -np.log1p(-alpha1)
            x2 = -np.log(alpha2-1)
        elif (alpha1 > 1) & (alpha2>1):
            x1 = -np.log(alpha1-1)
            x2 = -np.log(alpha2-1)
        else:
            raise ValueError
        
        return x1+x2+x3
        
    
    
    def sample(self, nSamples, lambdaBBHmass, mMin=5, mMax=100):
        alpha1, alpha2, beta, deltam, ml, mh, b = lambdaBBHmass
        
        
        pm1 = lambda x: np.exp(self._logpdfm1(x, alpha1, alpha2, deltam, ml, mh, b ))
        pm2 = lambda x: np.exp(self._logpdfm2(x,  beta, deltam, ml ))
        
        mMax = mh*(1+1/10)
        m1 = self._sample_pdf(nSamples, pm1, ml, mMax)
        m2 = self._sample_vector_upper(pm2, ml, m1)
        assert(m2<=m1).all() 
        assert(m2>=ml).all() 
        assert(m1<=mh).all() 
            
        return m1, m2


    
    
    
##############################################################################
# Power law + Gaussian peak


class PowerLawPlusPeakMass(BBHDistFunction):
    
    '''
    Mass distribution  - Truncated Power Law
    '''

    def __init__(self):
        
        BBHDistFunction.__init__(self)
        
        self.params = ['lambdaPeak', 'alpha', 'beta','deltam', 'ml', 'mh', 'muMass', 'sigmaMass' ]
        
        self.baseValues = {
                           'lambdaPeak':0.03,
                           'beta':0.81,
                           'alpha': 3.78 , 
                           'deltam':4.8,
                           'ml':5.,
                           'mh':112.,
                           'muMass': 32.,
                           'sigmaMass':3.88 
                           }
        
        self.names = { 'alpha': r'$\alpha$',
                           'beta':  r'$\beta_q$',
                           'deltam': r'$\delta_{\rm m}$',
                           'ml': r'$m_{\rm min}$',
                           'mh':r'$m_{\rm max}$',
                           'lambdaPeak': r'$\lambda_g$',
                           'muMass':r'$\mu_g$',
                           'sigmaMass':r'$\sigma_g$'
                           }
         
        self.n_params = len(self.params)
        
        print(' Power law + peak mass function base values: %s' %self.baseValues)
        
        
    def _logS(self, m, deltam, ml,):
        #print('_logS call')
        #print('Input m shape: %s' %str(m.shape))
        maskL = m <= ml 
        maskU = m >= (ml + deltam) 
        s = np.zeros(m.shape)
        s[maskL] = np.NINF
        s[maskU] = 0
        maskM = ~(maskL | maskU)
        s[maskM] = -np.logaddexp( 0, (deltam/(m[maskM]-ml) + deltam/(m[maskM]-ml - deltam) ) ) #1/(np.exp(deltam/(m[maskM]-ml) + deltam/(m[maskM]-ml - deltam))+1)
        #return np.where( maskM, -np.logaddexp( 0, (deltam/(m-ml) + deltam/(m-ml - deltam) ) ), np.NINF)
        #print('Output s shape: %s' %str(s.shape))
        return s
    
    def _logpdfm1(self, m, lambdaPeak, alpha, deltam, ml, mh, muMass, sigmaMass):
        '''
        Marginal distribution p(m1), not normalised
        '''

        #print('_logpdfm1 call')
        #print('Input m shape: %s' %str(m.shape))
        
        where_nan = np.isnan(m)
        result = np.zeros(m.shape)
        
        result[where_nan]=np.NINF
        
        max_compute = max(mh, muMass+10*sigmaMass)
        
        where_compute = (m <= max_compute) & (m >= ml) & (~where_nan)
        result[~where_compute] = np.NINF
        
        m = m[where_compute]
        trunc_component = np.exp(truncated_power_law(m, alpha, ml, mh)-norm_truncated_pl(alpha, ml, mh))
        gauss_component = np.exp(-(m-muMass)**2/(2*sigmaMass**2))/(np.sqrt(2*np.pi)*sigmaMass)
        
        result[where_compute] = np.log( (1-lambdaPeak)*trunc_component+lambdaPeak*gauss_component )+self._logS(m, deltam, ml)  #np.where(m < mBreak, np.log(m)*(-alpha1)+self._logS(m, deltam, ml), np.log(mBreak)*(-alpha1+alpha2)+np.log(m)*(-alpha2)+self._logS(m, deltam, ml) )
        
        #result = np.where( where_compute, np.log( (1-lambdaPeak)*( np.exp(truncated_power_law(m, alpha, ml, mh)-norm_truncated_pl(alpha, ml, mh))  ) +lambdaPeak*( np.exp(-(m-muMass)**2/(2*sigmaMass**2))/(np.sqrt(2*np.pi)*sigmaMass) ) )+self._logS(m, deltam, ml) , np.NINF)
        
        #print('_logpdfm1 call')
        #print('Output result shape: %s' %str(result.shape))
        return result
        
    
    
    def _logpdfm2(self, m2, beta, deltam, ml):
        '''
        Conditional distribution p(m2 | m1)
        '''
        
        #print('_logpdfm2 call')
        #print('Input m shape: %s' %str(m2.shape))
        
        where_nan = np.isnan(m2)
        result = np.zeros(m2.shape)

        result[where_nan]=np.NINF
        
        where_compute = (ml<= m2) & (~where_nan)
        result[~where_compute] = np.NINF
        
        m2 = m2[where_compute]
        result[where_compute] = np.log(m2)*(beta)+self._logS(m2, deltam, ml)
        #print('Out res shape: %s' %str(result.shape))
        #result = np.where( where_compute, np.log(m2)*(beta)+self._logS(m2, deltam, ml), np.NINF)
        return result
        
    
    
    def logpdf(self, theta, lambdaBBHmass, **kwargs):
        
        '''p(m1, m2 | Lambda ), normalized to one'''
        
        
        
        m1, m2 = theta
        lambdaPeak, alpha, beta, deltam, ml, mh, muMass, sigmaMass = lambdaBBHmass
        
        #print('logpdf call')
        #print('Input m1 shape: %s' %str(m1.shape))
        
        where_nan = np.isnan(m1)
        assert (where_nan==np.isnan(m2)).all()
        result = np.zeros(m1.shape)

        result[where_nan]=np.NINF
        
        max_compute = max(mh, muMass+10*sigmaMass)
        where_compute = (m2 < m1) & (ml< m2) & (m1 < max_compute ) & (~where_nan)
        
        #print('m1 logp(m1, m2): %s' %m1)
        #print('m2 logp(m1, m2): %s' %m2)
        #print('where_compute logp(m1, m2): %s' %where_compute)
        
        
        result[~where_compute] = np.NINF
        
        m1 = m1[where_compute]
        m2 = m2[where_compute]
        
        result[where_compute] = self._logpdfm1(m1,  lambdaPeak, alpha, deltam, ml, mh, muMass, sigmaMass ) + self._logpdfm2(m2, beta, deltam, ml) + self._logC(m1, beta, deltam,  ml, **kwargs)-  self._logNorm( lambdaPeak, alpha, deltam, ml, mh, muMass, sigmaMass )
        
        #result = np.where( where_compute, self._logpdfm1(m1,  lambdaPeak, alpha, deltam, ml, mh, muMass, sigmaMass ), np.NINF)
        
        #print('Out res shape logpdf: %s\n' %str(result.shape))
        
        return result
        
        
        
    
    
    def _logC(self, m, beta, deltam, ml, res = 200, exact_th=0.):
        '''
        Gives inverse log integral of  p(m1, m2) dm2 (i.e. log C(m1) in the LVC notation )
        '''
        
        #print('_logC call')
        #print('Input m1 shape: %s' %str(m.shape))
        
        #print(m)
        #print(ml+deltam+deltam/10+1e-01)
        #print(m[~np.isnan(m)].max())
        
        
        xlow=np.linspace(ml, ml+deltam+deltam/10, 200)
        xup=np.linspace(ml+deltam+deltam/10+1e-01, m[~np.isnan(m)].max(), res)
        xx=np.sort(np.concatenate([xlow,xup], ))
  
        p2 = np.exp(self._logpdfm2( xx , beta, deltam, ml))
        cdf = cumtrapz(p2, xx)
        
        where_compute = ~np.isnan(m)
        #where_exact = m <exact_th*ml
        
        #where_approx = (~where_exact) & (where_compute)
        
        
        result = np.zeros(m.shape)
        
        result[~where_compute]=np.NINF
        result[where_compute] = -np.log( np.interp(m[where_compute], xx[1:], cdf) )
        
        #result = np.where(where_compute, -np.log( np.interp(m, xx[1:], cdf) ) , np.NINF)
        #print('Out res shape: %s' %str(result.shape))
        return result
    
        


    def _logNorm(self, lambdaPeak, alpha, deltam, ml, mh, muMass, sigmaMass  , res=200):
        '''
        Gives log integral of  p(m1, m2) dm1 dm2 (i.e. total normalization of mass function )

        '''

        if lambdaPeak==0:
            ms = np.linspace(ml, mh, res) 
        else:
            max_compute = max(mh, muMass+10*sigmaMass)
            
            # lower edge
            ms1 = np.linspace(1., ml+deltam+deltam/10, 200)
            
            # before gaussian peak
            ms2 = np.linspace( ml+deltam+deltam/10+1e-01, muMass-5*sigmaMass, int(res/2) )
            
            # around gaussian peak
            ms3= np.linspace( muMass-5*sigmaMass+1e-01, muMass+5*sigmaMass, int(res/2) )
            
            # after gaussian peak
            ms4 = np.linspace(muMass+5*sigmaMass+1e-01, max_compute+max_compute/2, int(res/2) )
            
            ms=np.sort(np.concatenate([ms1,ms2, ms3, ms4], ))
        
        p1 = np.exp(self._logpdfm1( ms , lambdaPeak, alpha, deltam, ml, mh, muMass, sigmaMass  ))
        return np.log(np.trapz(p1,ms))
    
    
    def sample(self, nSamples, lambdaBBHmass, mMin=2., mMax=200.):
        lambdaPeak, alpha, beta, deltam, ml, mh, muMass, sigmaMass = lambdaBBHmass
        
        
        pm1 = lambda x: np.exp(self._logpdfm1(x, lambdaPeak, alpha, deltam, ml, mh, muMass, sigmaMass ))
        pm2 = lambda x: np.exp(self._logpdfm2(x,  beta, deltam, ml ))
        
        mMax = max(mh, muMass+5*sigmaMass)
        m1 = self._sample_pdf(nSamples, pm1, ml, mMax)
        m2 = self._sample_vector_upper(pm2, ml, m1)
        assert(m2<=m1).all() 
        assert(m2>=ml).all() 
        assert(m1<=mh).all() 
            
        return m1, m2
    
    
       
    
#######################################################
###################    MULTIPEAK    ###################
#######################################################
    
    
class MultiPeakMass(BBHDistFunction):
    
    '''
    Mass distribution  - Multipeak Power Law
    '''

    def __init__(self):
        
        BBHDistFunction.__init__(self)
        
        self.params = ['lambdaPeak', 'lambda1', 'alpha', 'beta', 'ml', 'mh', 'mu1', 'sigma1', 'mu2', 'sigma2', 'deltam' ]
        
        self.baseValues = {
                           'lambdaPeak':0.05,
                           'lambda1':0.5,
                           'alpha': 2.9,
                           'beta':0.9,
                           'ml':4.6,
                           'mh':87.,
                           'mu1': 33.,
                           'sigma1':3.,
                           'mu2': 68.,
                           'sigma2':3.,                          
                           'deltam':4.8
                           }
        
        self.names = { 'lambdaPeak': r'$\lambda$',
                       'lambda1': r'$\lambda_1$',
                       'alpha': r'$\alpha$',
                       'beta':  r'$\beta_q$',
                       'ml': r'$m_{\rm min}$',
                       'mh':r'$m_{\rm max}$',
                       'mu1':r'$\mu_1$',
                       'sigma1':r'$\sigma_1$',
                       'mu2':r'$\mu_2$',
                       'sigma2':r'$\sigma_2$',
                       'deltam': r'$\delta_{\rm m}$'
                       }
         
        self.n_params = len(self.params)
        
        print(' Multi peak mass function base values: %s' %self.baseValues)
        
        
    def _logS(self, m, deltam, ml,):
        maskL = m <= ml #- eps
        maskU = m >= (ml + deltam) #+ eps
        s = np.zeros(m.shape)
        s[maskL] = np.NINF
        s[maskU] = 0
        maskM = ~(maskL | maskU)
        s[maskM] = -np.logaddexp( 0, (deltam/(m[maskM]-ml) + deltam/(m[maskM]-ml - deltam) ) ) #1/(np.exp(deltam/(m[maskM]-ml) + deltam/(m[maskM]-ml - deltam))+1)
        return s
    
    
    def _logpdfm1(self, m, lambdaPeak, lambda1, alpha, ml, mh, mu1, sigma1, mu2, sigma2, deltam):
        '''
        Marginal distribution p(m1), not normalised
        '''

        where_nan = np.isnan(m)
        result = np.empty_like(m)
        
        result[where_nan]=np.NINF
        
        max_compute = max(mh, mu2+10*sigma2)
        
        where_compute = (m <= max_compute) & (m >= ml) & (~where_nan)
        result[~where_compute] = np.NINF
        
        m = m[where_compute]
        trunc_component = np.exp(truncated_power_law(m, alpha, ml, mh)-norm_truncated_pl(alpha, ml, mh))
        gauss_component1 = np.exp(-(m-mu1)**2/(2*sigma1**2))/(np.sqrt(2*np.pi)*sigma1)
        gauss_component2 = np.exp(-(m-mu2)**2/(2*sigma2**2))/(np.sqrt(2*np.pi)*sigma2)
        
        result[where_compute] = np.log((1-lambdaPeak)*trunc_component+lambdaPeak*lambda1*gauss_component1 + lambdaPeak*(1-lambda1)*gauss_component2  ) + self._logS(m, deltam, ml)  #np.where(m < mBreak, np.log(m)*(-alpha1)+self._logS(m, deltam, ml), np.log(mBreak)*(-alpha1+alpha2)+np.log(m)*(-alpha2)+self._logS(m, deltam, ml) )
        
        return result
        
    
    
    def _logpdfm2(self, m2, beta, deltam, ml):
        '''
        Conditional distribution p(m2 | m1)
        '''
        where_nan = np.isnan(m2)
        result = np.empty_like(m2)

        result[where_nan]=np.NINF
        
        where_compute = (ml<= m2) & (~where_nan)
        result[~where_compute] = np.NINF
        
        m2 = m2[where_compute]
        result[where_compute] = np.log(m2)*(beta)+self._logS(m2, deltam, ml)
        return result
        
    
    
    def logpdf(self, theta, lambdaBBHmass):
        
        '''p(m1, m2 | Lambda ), normalized to one'''
        
        m1, m2 = theta
        lambdaPeak, lambda1, alpha, beta, ml, mh, mu1, sigma1, mu2, sigma2, deltam = lambdaBBHmass
        
        where_nan = np.isnan(m1)
        assert (where_nan==np.isnan(m2)).all()
        result = np.empty_like(m1)

        result[where_nan]=np.NINF
        
        max_compute = max(mh, mu2+10*sigma2)
        where_compute = (m2 < m1) & (ml< m2) & (m1 < max_compute ) & (~where_nan)
        result[~where_compute] = np.NINF
        
        m1 = m1[where_compute]
        m2 = m2[where_compute]
        
        result[where_compute] = self._logpdfm1(m1, lambdaPeak, lambda1, alpha, ml, mh, mu1, sigma1, mu2, sigma2, deltam) + self._logpdfm2(m2, beta, deltam, ml) + self._logC(m1, beta, deltam,  ml) - self._logNorm(lambdaPeak, lambda1, alpha, ml, mh, mu1, sigma1, mu2, sigma2, deltam)
        
        return result
        
        
        
    
    
    def _logC(self, m, beta, deltam, ml):
        '''
        Gives inverse log integral of  p(m1, m2) dm2 (i.e. log C(m1) in the LVC notation )
        '''
        res = 200
        exact_th=0.
        
        xlow=np.linspace(ml, ml+deltam+deltam/10, 200)
        xup=np.linspace(ml+deltam+deltam/10+1e-01, m[~np.isnan(m)].max(), res)
        xx=np.sort(np.concatenate([xlow,xup], ))
  
        p2 = np.exp(self._logpdfm2( xx , beta, deltam, ml))
        cdf = cumtrapz(p2, xx)
        
        where_compute = ~np.isnan(m)
        #where_exact = m <exact_th*ml
        
        #where_approx = (~where_exact) & (where_compute)
        
        result = np.empty_like(m)
        
        result[~where_compute]=np.NINF
        #result[where_exact]=self._logCexact(m[where_exact], beta, deltam, ml,) #np.NINF
        
        result[where_compute] = -np.log( np.interp(m[where_compute], xx[1:], cdf) )
        
        return result
    
        


    def _logNorm(self, lambdaPeak, lambda1, alpha, ml, mh, mu1, sigma1, mu2, sigma2, deltam ):
        '''
        Gives log integral of  p(m1, m2) dm1 dm2 (i.e. total normalization of mass function )

        '''
        res=200
        
        if lambdaPeak==0 and lambda1==0:
            ms = np.linspace(ml, mh, res) 
        else:
            max_compute = max(mh, mu2+10*sigma2)
            
            # lower edge
            ms1 = np.linspace(1., ml+deltam+deltam/10, 200)
            
            # before first gaussian peak
            ms2 = np.linspace( ml+deltam+deltam/10+1e-01, mu1-5*sigma1, int(res/2) )
            
            # around first gaussian peak
            ms3 = np.linspace( mu1-5*sigma1+1e-01, mu1+5*sigma1, int(res/2) )
            
            # after first gaussian peak, before second gaussian peak
            ms4 = np.linspace( mu1+5*sigma1+1e-01, mu2-5*sigma2, int(res/2) )
            
            # around second gaussian peak
            ms5 = np.linspace( mu2-5*sigma2+1e-01, mu2+5*sigma2, int(res/2) )
            
            # after second gaussian peak
            ms6 = np.linspace(mu2+5*sigma2+1e-01, max_compute+max_compute/2, int(res/2) )
            
            ms=np.sort(np.concatenate([ms1,ms2, ms3, ms4, ms5, ms6], ))
        
        p1 = np.exp(self._logpdfm1( ms , lambdaPeak, lambda1, alpha, ml, mh, mu1, sigma1, mu2, sigma2, deltam  ))
        return np.log(np.trapz(p1,ms))
    
    
    def sample(self, nSamples, lambdaBBHmass, mMin=2., mMax=200.):
        
        lambdaPeak, lambda1, alpha, beta, ml, mh, mu1, sigma1, mu2, sigma2, deltam = lambdaBBHmass
        
        pm1 = lambda x: np.exp(self._logpdfm1(x, lambdaPeak, lambda1, alpha, ml, mh, mu1, sigma1, mu2, sigma2, deltam ))
        pm2 = lambda x: np.exp(self._logpdfm2(x,  beta, deltam, ml ))
        
        mMax = max(mh, mu2+10*sigma2)
        m1 = self._sample_pdf(nSamples, pm1, ml, mMax)
        m2 = self._sample_vector_upper(pm2, ml, m1)
        assert(m2<=m1).all() 
        assert(m2>=ml).all() 
        assert(m1<=mh).all() 
            
        return m1, m2

    
    

##############################################################################
##############################################################################
##############################################################################
# Neutron stars here


class BNSGaussMass(BBHDistFunction):
    
    '''
    BNS Mass distribution  - Gaussian in the two masses, uncorrelated
    '''
    
    def __init__(self, ):
        
        BBHDistFunction.__init__(self)
        
        self.params = ['meanMass', 'sigmaMass']
        
        self.baseValues = {
                           
                           'meanMass':1.4,
                           'sigmaMass':0.15, 
                           }
        
        self.names = {
                           'meanMass':r'$\mu_{\rm mass}$',
                           'sigmaMass':r'$\sigma_{\rm mass}$', 
                           }
         
        self.n_params = len(self.params)
    
        
    
    
    def logpdf(self, theta, lambdaBBHmass):
        
        '''p(m1, m2 | Lambda ) = p(m1)*p(m2), normalized to one'''
        
        m1, m2 = theta
        meanMass, sigmaMass = lambdaBBHmass
        
        #condition = (m2<m1) & (m2>5)
        #if not condition:
        #    return np.NINF
    
        #logpdfMass = -2*np.log(sigmaMass)-(m1-meanMass)**2/(2*sigmaMass**2)-np.log(2*np.pi)-(m2-meanMass)**2/(2*sigmaMass**2)
        
        logpdfm1 = np.log(truncGausslower(m1, 0., loc=meanMass, scale=sigmaMass))
        logpdfm2 = np.log(truncGausslower(m2, 0., loc=meanMass, scale=sigmaMass))
        
        return logpdfm1+logpdfm2
    
    
    def sample(self, nSamples, lambdaBBHmass, ):
        
        mu, sig = lambdaBBHmass
        
        #m1s = np.random.normal(loc=mu, scale=sig, size=nSamples)
        #m2s = np.random.normal(loc=mu, scale=sig, size=nSamples)
        
        m1s = sample1d(nSamples, lambda x: truncGausslower(x, 0., loc=mu, scale=sig), max(0, mu-10*sig), mu+10*sig, )
        m2s = sample1d(nSamples, lambda x: truncGausslower(x, 0., loc=mu, scale=sig), max(0, mu-10*sig), mu+10*sig, )
        
        m1 = np.where(m1s>m2s, m1s, m2s)
        m2 = np.where(m1s>m2s, m2s, m1s)
        
        return m1, m2


def sample1d(nSamples, pdf, lower, upper, res = 100000):

    ''' Sample from continuous pdf within bounds ''' 

    # this is about 15% faster than using a sample_discrete like approach to sample the grid, and allows for returning continuous samples 
    x = np.linspace(lower, upper, res)
    cdf = np.cumsum(pdf(x))
    cdf = cdf / cdf[-1]
    return np.interp(np.random.uniform(size=nSamples), cdf, x)

def sampleMultipleTruncGaussLow(locs, scales, lowers):
    from scipy.special import erf, erfinv
    
    Phialphas = 0.5*(1.+erf((lowers-locs)/(np.sqrt(2.)*scales)))
    unifSam = np.random.uniform(size=len(locs))
    arg = Phialphas + unifSam*(1.-Phialphas)
    return np.sqrt(2)*erfinv(2.*arg - 1.)*scales + locs


def truncGausslower(x, xmin, loc=0., scale=1.):
    from scipy.special import erf

    Phialpha = 0.5*(1.+erf((xmin-loc)/(np.sqrt(2.)*scale)))
    return np.where(x>xmin, 1./(np.sqrt(2.*np.pi)*scale)/(1.-Phialpha) * np.exp(-(x-loc)**2/(2*scale**2)) ,0.)



class BNSFlatMass(BBHDistFunction):
    
    '''
    BNS Mass distribution  - Flat in the two masses, uncorrelated
    '''
    
    def __init__(self, ):
        
        BBHDistFunction.__init__(self)
        
        self.params = ['ml', 'mh']
        
        self.baseValues = {
                           
                           'ml':1.,
                           'mh':3., 
                           }
        
        self.names = {
                           'ml':r'$\m_{\rm min}$',
                           'mh':r'$\m_{\rm max}$', 
                           }
         
        self.n_params = len(self.params)
    
        
    
    
    def logpdf(self, theta, lambdaBBHmass):
        
        '''p(m1, m2 | Lambda ) = p(m1)*p(m2), normalized to one'''
        
        m1, m2 = theta
        ml, mh = lambdaBBHmass
        lognorm = -np.log(mh-ml)
        
        logpdf1 = np.full(m1.shape, np.NINF)
        mask1 = (m1>ml) & (m1<mh)
        logpdf1[mask1] = lognorm
        
        logpdf2 = np.full(m2.shape, np.NINF)
        mask2 = (m2>ml) & (m2<mh)
        logpdf2[mask2] = lognorm

        return logpdf1+logpdf2
    
    
    def sample(self, nSamples, lambdaBBHmass, ):
        
        ml, mh = lambdaBBHmass
        
        m1s = np.random.uniform(low=ml, high=mh, size=nSamples)
        m2s = np.random.uniform(low=ml, high=mh, size=nSamples)
        
        m1 = np.where(m1s>m2s, m1s, m2s)
        m2 = np.where(m1s>m2s, m2s, m1s)
        
        return m1, m2





##############################################################################
# Broken power law for BNS


class BrokenPowerLawMassBNS(BBHDistFunction):
    
    '''
    Mass distribution  - Broken Power Law
    '''

    def __init__(self):
        
        BBHDistFunction.__init__(self)
        
        self.params = ['alpha1', 'alpha2',  'deltam', 'ml',  'mh', 'b' ]
        
        self.baseValues = {
                           'alpha1':1.6,
                           'alpha2': 5.6 , 
                           #'beta': 1.4,
                           'deltam':4.8,
                           'ml':4,
                           'mh':87,
                           'b': 0.43,
                           }
        
        self.names = { 'alpha1': r'$\alpha_1$',
                           'alpha2':  r'$\alpha_2$' , 
                           #'beta':  r'$\beta_q$',
                           'deltam': r'$\delta_{\rm m}$',
                           'ml': r'$m_{\rm min}$',
                           'mh':r'$m_{\rm max}$',
                           'b': r'$b$',
                           
                           }
         
        self.n_params = len(self.params)
        
        print('Broken power law mass function base values: %s' %self.baseValues)
    
    
    def _get_Mbreak(self, mMin, mMax, b):
        return  mMin + b*(mMax - mMin)
    
    
    def _logS(self, m, deltam, ml,):
        maskL = m <= ml #- eps
        maskU = m >= (ml + deltam) #+ eps
        s = np.empty_like(m)
        s[maskL] = np.NINF
        s[maskU] = 0
        maskM = ~(maskL | maskU)
        s[maskM] = -np.logaddexp( 0, (deltam/(m[maskM]-ml) + deltam/(m[maskM]-ml - deltam) ) ) #1/(np.exp(deltam/(m[maskM]-ml) + deltam/(m[maskM]-ml - deltam))+1)
        return s
    
    
    def _logpdfm1(self, m,  alpha1, alpha2, deltam, ml, mh, b):
        '''
        Marginal distribution p(m1), not normalised
        '''

        mBreak = self._get_Mbreak( ml, mh, b)
        
        where_nan = np.isnan(m)
        result = np.empty_like(m)
        
        result[where_nan]=np.NINF
        
        where_compute = (m <= mh) & (m >= ml) & (~where_nan)
        result[~where_compute] = np.NINF
        
        m = m[where_compute]
        result[where_compute] = np.where(m < mBreak, np.log(m)*(-alpha1)+self._logS(m, deltam, ml), np.log(mBreak)*(-alpha1+alpha2)+np.log(m)*(-alpha2)+self._logS(m, deltam, ml) )
        
        return result
        
    

        
    
    
    def logpdf(self, theta, lambdaBBHmass, **kwargs):
        
        '''p(m1, m2 | Lambda ), normalized to one'''
        
        m1, m2 = theta
        alpha1, alpha2,  deltam, ml, mh, b = lambdaBBHmass
        where_nan = np.isnan(m1)
        
        logpdf1 = np.full(m1.shape, np.NINF)
        mask1 = (m1>ml) & (m1<mh) & (~where_nan)
        logpdf1[mask1] = self._logpdfm1(m1[mask1],  alpha1, alpha2, deltam, ml, mh, b )
        
        logpdf2 = np.full(m2.shape, np.NINF)
        mask2 = (m2>ml) & (m2<mh) & (~where_nan)
        logpdf2[mask2] = self._logpdfm1(m2[mask2],  alpha1, alpha2, deltam, ml, mh, b )

        return logpdf1+logpdf2-2*self._logNorm( alpha1, alpha2, deltam, ml, mh, b,)
        
        
        #result[where_compute] = self._logpdfm1(m1,  alpha1, alpha2, deltam, ml, mh, b ) + self._logpdfm2(m2, beta, deltam, ml) + self._logC(m1, beta, deltam,  ml, **kwargs)-  self._logNorm( alpha1, alpha2, deltam, ml, mh, b,)
        #return result
        


    def _logNorm(self, alpha1, alpha2, deltam, ml, mh, b , res=200):
        '''
        Gives log integral of  p(m1, m2) dm1 dm2 (i.e. total normalization of mass function )

        '''
        
        mbr = self._get_Mbreak( ml, mh, b)
        

        ms1 = np.linspace(1., ml+deltam+deltam/10, 200)
        ms2 = np.linspace( ml+deltam+deltam/10+1e-01, mbr-mbr/10, int(res/2) )
        ms3= np.linspace( mbr-mbr/10+1e-01, mbr+mbr/10, 50 )
        ms4 = np.linspace(mbr+mbr/10+1e-01, mh+mh/10, int(res/2) )
        
        ms=np.sort(np.concatenate([ms1,ms2, ms3, ms4], ))
        
        p1 = np.exp(self._logpdfm1( ms ,alpha1, alpha2, deltam, ml, mh, b ))
        return np.log(np.trapz(p1,ms))
        

        
    
    
    def sample(self, nSamples, lambdaBBHmass, mMin=5, mMax=100):
        alpha1, alpha2, deltam, ml, mh, b = lambdaBBHmass
        
        
        pm1 = lambda x: np.exp(self._logpdfm1(x, alpha1, alpha2, deltam, ml, mh, b ))
        
        mMax = mh*(1+1/10)
        m1s = self._sample_pdf(nSamples, pm1, ml, mMax)
        m2s = self._sample_pdf(nSamples, pm1, ml, mMax)
        
        m1 = np.where(m1s>m2s, m1s, m2s)
        m2 = np.where(m1s>m2s, m2s, m1s)
        
        assert(m2<=m1).all() 
        assert(m2>=ml).all() 
        assert(m1<=mh).all() 
            
        return m1, m2



##############################################################################
# Dummy class; for testing


class AstroGaussMass(BBHDistFunction):
    
    '''
    Mass distribution  - Gaussian in the first mass flat in the second
    '''
    
    def __init__(self, ):
        
        BBHDistFunction.__init__(self)
        
        self.params = ['meanMass', 'sigmaMass']
        
        self.baseValues = {
                           
                           'meanMass':20,
                           'sigmaMass':10, 
                           }
        
        self.names = {
                           'meanMass':r'$\mu_{\rm mass}$',
                           'sigmaMass':r'$\sigma_{\rm mass}$', 
                           }
         
        self.n_params = len(self.params)
    
        
    
    
    def logpdf(self, theta, lambdaBBHmass):
        
        '''p(m1, m2 | Lambda ), normalized to one'''
        
        m1, m2 = theta
        meanMass, sigmaMass = lambdaBBHmass
        
        condition = (m2<m1) & (m2>5)
        if not condition:
            return np.NINF
    
        logpdfMass = (-np.log(sigmaMass)-(m1-meanMass)**2/(2*sigmaMass**2)) 
        
        return logpdfMass
    
    
