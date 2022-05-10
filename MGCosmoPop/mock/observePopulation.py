#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 13:47:59 2021

@author: Michi
"""


import numpy as np
#import random
import h5py
import os 
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

#from scipy.interpolate import interp1d
#from scipy.integrate import cumtrapz

#from .SNRtools import oSNR, NetworkSNR, Detector
#import scipy.stats as ss

#import Globals


from scipy.special import erfc, erfcinv

def sample_trunc_gaussian(mu = 1, sigma = 1, lower = 0, size = 1):

    sqrt2 = np.sqrt(2)
    Phialpha = 0.5*erfc(-(lower-mu)/(sqrt2*sigma))
    
    if np.isscalar(mu):
        arg = Phialpha + np.random.uniform(size=size)*(1-Phialpha)
        return np.squeeze(mu - sigma*sqrt2*erfcinv(2*arg))
    else:
        Phialpha = Phialpha[:,np.newaxis]
        arg = Phialpha + np.random.uniform(size=(mu.size, size))*(1-Phialpha)
        
        return np.squeeze(mu[:,np.newaxis] - sigma[:,np.newaxis]*sqrt2*erfcinv(2*arg))




class Observations(object):
    
    
    def __init__(self,
                 populations, 
                 detNet,
                 zmax, out_dir,
                 rho_th=8., eta_scatter=5e-03, mc_scatter = 3e-02, theta_scatter = 5e-02,
#                 **kwargs):
    ):        
        
        self.out_dir=out_dir
        
        #self.osnr = oSNR(**kwargs) 
        #self.osnr.make_interpolator()
        
        
        ## Define detector network
        self.detNet = detNet #NetworkSNR()
        
        
        self.allPops=populations
        
        self.zmax = zmax #self.find_max_z()
        
        self.rho_th=rho_th
        self.eta_scatter=eta_scatter
        self.mc_scatter=mc_scatter
        self.theta_scatter=theta_scatter
    
        
        
        self.lambdaBase = self.allPops.get_base_values(self.allPops.params)
        self.LambdaCosmoBase, self.LambdaAllPopBase = self.allPops._split_params(self.lambdaBase)
        self.H0base, self.Om0Base, self.w0Base, self.Xi0Base, self.nBase = self.allPops.cosmo._get_values(self.LambdaCosmoBase, [ 'H0', 'Om', 'w0', 'Xi0', 'n'])
    
#        self._get_theta_cdf()
        
        self._find_Nperyear_expected()
        
    def _find_Nperyear_expected(self):  
         self.Nperyear_expected = self.allPops.Nperyear_expected(self.lambdaBase, zmax=self.zmax, verbose=True)
         print('Expected number per year between redshift 0 and %s: %s'%(self.zmax, self.Nperyear_expected) )

    
    def _find_Nperyear_expected_old(self):
        LambdaPop = self.LambdaAllPopBase[0:self.allPops._allNParams[0]]
        lambdaBBHrate, lambdaBBHmass, lambdaBBHspin = self.allPops._pops[0]._split_lambdas(LambdaPop)
        
        zz = np.linspace(0, self.zmax, 1000)
        
        print('rate parameters in _find_Nperyear_expected: %s' %str(lambdaBBHrate))
        dNdz = np.exp(self.allPops.logdN_dz( zz, self.H0base, self.Om0Base, self.w0Base, lambdaBBHrate, self.allPops._pops[0]))
        
        self.Nperyear_expected = np.trapz(dNdz,zz)
        
        print('Expected number per year between redshift 0 and %s: %s'%(self.zmax, self.Nperyear_expected) )
    
    
    def  _generate_mergers(self, N, verbose=True):
        
        if N==1:
            verbose=False
        if verbose:
            print('Parameters used to sample events: %s' %str(self.lambdaBase))
        massSamples, zs, spinSamples = self.allPops.sample(N, self.zmax, self.lambdaBase)
        #m1s, m2s, zs = allSamples[:, :, 0], allSamples[:, :, 1], allSamples[:, :, 2] 
        m1s, m2s = np.squeeze(massSamples)[:, 0], np.squeeze(massSamples)[:, 1]
    
        costhetas = 1.-2.*np.random.uniform(size=N)
        phis = 2.*np.pi*np.random.uniform(size=N)
        cosiotas = 1.-2.*np.random.uniform(size=N)
        ts_det = np.random.uniform(size=N)

        
        return np.squeeze(m1s), np.squeeze(m2s), np.squeeze(zs), costhetas,  phis, cosiotas, ts_det
    
    
    
    def _generate_injection_chunk(self,  N,  seed=1312, ):
        
        print('Generating events in source frame and redshift...')
        m1s, m2s, zs, costhetas,  phis, cosiotas, ts_det = self._generate_mergers(N)
        print('Generated %s events .' %str(m1s.shape))
        
        #keep = np.random.rand(N)<duty_cycle
            
        #m1s, m2s, zs, thetas = m1sGen[keep], m2sGen[keep], zsGen[keep], thetasGen[keep]
        
        #print('%s events kept assuming %s duty cycle' %(m1s.shape[0], duty_cycle))
        
        
        ## Get quantities in detector frame
        
        dLs = self.allPops.cosmo.dLGW(zs, self.H0base, self.Om0Base, self.w0Base, self.Xi0Base, self.nBase)
        m1d, m2d, = m1s*(1+zs), m2s*(1+zs)
        
        ## Get SNR
        #oSNRs = self.osnr.get_oSNR(m1d, m2d, dLs)
        #SNR = oSNRs*thetas
        
        SNR = self.detNet.full_SNR(m1d, m2d, dLs, costhetas, phis, cosiotas, ts_det)
 
        rho_obs = SNR + np.random.randn(SNR.shape[0])
        
        out = rho_obs<0
        print('Imposing observed SNR>0...')
        while np.any(out):
            rho_obs[out] = SNR[out]+np.random.randn(rho_obs[out].shape[0])
            out = rho_obs<0
        
        
        # Get p_draw
        #LambdaPop = self.LambdaAllPopBase[0:self.allPops._allNParams[0]]
        #lambdaBBHrate, lambdaBBHmass, lambdaBBHspin = self.allPops._pops[0]._split_lambdas(LambdaPop)
                
        #logdNdz = self.allPops.logdN_dz( zs, self.H0base, self.Om0Base, self.w0Base, lambdaBBHrate, self.allPops._pops[0])-np.log(self.Nperyear_expected)
        
        #logpm1m2 = self.allPops._pops[0].massDist.logpdf([m1s, m2s], lambdaBBHmass)
        

        log_p_draw = self.allPops.log_dN_dm1zdm2zddL(m1s, m2s, zs, [], 1., self.lambdaBase, dL=dLs)-np.log(self.Nperyear_expected) #logdNdz+logpm1m2-2*np.log1p(zs)-self.allPops.cosmo.log_ddL_dz(zs, self.H0base, self.Om0Base, self.w0Base, self.Xi0Base, self.nBase )
        #p_draw=np.exp(log_p_draw)
        
        # Select
        keep = rho_obs>self.rho_th
        
        
        return m1d[keep], m2d[keep], dLs[keep], log_p_draw[keep], rho_obs[keep]
        
        
    def generate_injections(self, N_goal, chunk_size=int(1e05), seed=1312):
        N_gen =0
        Nsucc=0
        m1s_det = []
        m2s_det = []
        dls_det = []
        logwts_det = []
        rhos_det = []
        enough=False
        while not enough:

            m1d, m2d, dls, logwts, rho = self._generate_injection_chunk( chunk_size, seed=seed)
    
            N_gen += chunk_size
    
            m1s_det.append(m1d)
            m2s_det.append(m2d)
            dls_det.append(dls)
            rhos_det.append(rho)
            logwts_det.append(logwts)
         
            Nsucc += len(m1d)
            print('Total kept: so far= %s' %Nsucc)
        
            if Nsucc > N_goal:
                enough=True
            
        m1s_det = np.concatenate(m1s_det)
        m2s_det = np.concatenate(m2s_det)
        dls_det = np.concatenate(dls_det)
        logwts_det = np.concatenate(logwts_det)
        rhos_det = np.concatenate(rhos_det)
        print('Total generated: %s' %N_gen)
        print('Total kept: %s' %Nsucc)
        
        with h5py.File(os.path.join(self.out_dir,'selected.h5'), 'w') as f:
            f.attrs['N_gen'] = N_gen
            f.attrs['snr_th'] = self.rho_th
            f.create_dataset('m1det', data=m1s_det, compression='gzip', shuffle=True)
            f.create_dataset('m2det', data=m2s_det, compression='gzip', shuffle=True)
            f.create_dataset('dl', data=dls_det, compression='gzip', shuffle=True)
            f.create_dataset('logwt', data=logwts_det, compression='gzip', shuffle=True)
            f.create_dataset('snr', data=rhos_det, compression='gzip', shuffle=True)
    
    
 
