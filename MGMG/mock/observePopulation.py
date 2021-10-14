#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 13:47:59 2021

@author: Michi
"""


import numpy as np
import random
import h5py
import os 
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz

from .SNRtools import oSNR
import scipy.stats as ss

import Globals


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
                 populations, zmax, out_dir,
                 rho_th=8., eta_scatter=5e-03, mc_scatter = 3e-02, theta_scatter = 5e-02,
                 **kwargs):
        
        
        self.out_dir=out_dir
        
        self.osnr = oSNR(**kwargs) 
        self.osnr.make_interpolator()
        
        self.allPops=populations
        
        self.zmax = zmax #self.find_max_z()
        
        self.rho_th=rho_th
        self.eta_scatter=eta_scatter
        self.mc_scatter=mc_scatter
        self.theta_scatter=theta_scatter
    
        
        
        self.lambdaBase = self.allPops.get_base_values(self.allPops.params)
        self.LambdaCosmoBase, self.LambdaAllPopBase = self.allPops._split_params(self.lambdaBase)
        self.H0base, self.Om0Base, self.w0Base, self.Xi0Base, self.nBase = self.allPops.cosmo._get_values(self.LambdaCosmoBase, [ 'H0', 'Om', 'w0', 'Xi0', 'n'])
    
        self._get_theta_cdf()
        
        self._find_Nperyear_expected()
        
        
    
    def _get_theta_cdf(self):
        thpath = os.path.join(Globals.thetaPath, 'thetas.h5')
        
        with h5py.File(thpath, 'r') as f:
            ts = np.array(f['Theta'])
            ts = np.concatenate((ts, [0.0, 1.0]))
            ts = np.sort(ts)
            
        kde = ss.gaussian_kde(ts)
        tts = np.linspace(0, 1, 1000)
        ps = kde(tts) + kde(-tts) + kde(2-tts)
        
        self.theta_p = interp1d(tts, ps)
        self.theta_cdf =  interp1d( ts, np.linspace(0, 1, ts.shape[0]))
        
    
    
    
    def _find_Nperyear_expected(self):
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
        allSamples = self.allPops.sample(N, self.zmax, self.lambdaBase)
        m1s, m2s, zs = allSamples[:, :, 0], allSamples[:, :, 1], allSamples[:, :, 2] 
    
        ## Get theta samples

        x = np.linspace(0,1, 10000)
        cdf = self.theta_cdf(x)
        thetas= np.interp(np.random.uniform(size=N), cdf, x)
        
        return np.squeeze(m1s), np.squeeze(m2s), np.squeeze(zs), np.squeeze(thetas)
    
    
    
    def _generate_observations(self, m1s, m2s, zs, theta, verbose=True, eps=0.001):
        
        
        if np.isscalar(m1s) or m1s.ndim==0:
            dim = 1
            verbose=False
        else:
            dim = m1s.shape[0]
        
        ## Get quantities in detector frame
        print('COSMO PARAMETERS FOR DL: %s' %str([self.H0base, self.Om0Base, self.w0Base, self.Xi0Base, self.nBase]))
        dLs = self.allPops.cosmo.dLGW(zs, self.H0base, self.Om0Base, self.w0Base, self.Xi0Base, self.nBase)
        m1d, m2d, = m1s*(1+zs), m2s*(1+zs)
        print('Example: z = %s, dLs: %s' %(str(zs[:5]), str(dLs[:5])) )
        oSNRs = self.osnr.get_oSNR(m1d, m2d, dLs)
        SNR = oSNRs*theta
        if verbose:
            print('N. of opt SNRs >th: %s' %str(oSNRs[oSNRs>self.rho_th].shape))
            print('N. of raw SNRs >th: %s' %str(SNR[SNR>self.rho_th].shape))
        
        
        
        rho_obs = SNR + np.random.randn(dim)
        out = rho_obs<0
        print('Imposing observed SNR>0...')
        while np.any(out):
            rho_obs[out] = SNR[out]+np.random.randn(rho_obs[out].shape[0])
            out = rho_obs<0
        
        sigma_rho = np.ones(dim)
    
        mtot = m1d+m2d
        eta = m1d*m2d/(mtot**2)
        mc = mtot*eta**(3.0/5.0)
    
        sigma_mc = abs( self.rho_th/rho_obs*self.mc_scatter)    
        mc_obs = np.random.lognormal(mean=np.log(mc), sigma=sigma_mc)
    
        sigma_eta = self.rho_th/rho_obs*self.eta_scatter
        sigma_theta = self.rho_th/rho_obs*self.theta_scatter
        #eta_obs = np.ones(m1s.shape)
        #while (eta_obs < 0) or (eta_obs > 0.25):
        
        if dim>1:      
            eta_obs = eta + sigma_eta*np.random.randn(dim)
            out = (eta_obs<0) | (eta_obs>0.25)
            print('Imposing cut on Mc...')
            #print(eta)
            #print(sigma_eta)
            npoints=0
            while np.any(out):
                replace =  eta[out] + sigma_eta[out]*np.random.randn(eta[out].shape[0])
                Nrep = out.sum()
                if Nrep!=npoints and Nrep<6:
                    print('N. of points to replace: %s' %str(Nrep))
                    npoints=Nrep
                    #if Nrep==1:
                    #print('eta, sigma eta, rho_obs, m1, m2, z: %s' %str([eta[out], sigma_eta[out], rho_obs[out], m1s[out], m2s[out], zs[out]]) )
                    replace =  np.random.uniform(low=0, high=0.25, size=1)
               
                eta_obs[out] = replace  
                out = (eta_obs<0) | (eta_obs>0.25)
            assert np.all( (eta_obs>0) & (eta_obs<0.25))
        
            
            theta_obs = theta + sigma_theta*np.random.randn(dim)
            out = (theta_obs<0) | (theta_obs>1)
            print('Imposing cut on Theta...')
            npoints=0
            while np.any(out):
                Nrep = out.sum()
                replace=theta[out] + sigma_theta[out]*np.random.randn(theta[out].shape[0])
                if Nrep!=npoints and Nrep<6:
                    
                    print('N. of points to replace: %s' %str(Nrep))
                    npoints=Nrep
                    #if Nrep==1:
                        #print('theta, sigma theta, rho_obs, m1, m2, z: %s' %str([theta[out], sigma_theta[out], rho_obs[out], m1s[out], m2s[out], zs[out]]) )
                    replace = np.random.uniform(low=0, high=1, size=1)
                    
                theta_obs[out] =  replace
                out = (theta_obs<0) | (theta_obs>1)
            assert np.all( (theta_obs>0) & (theta_obs<1))
        
        else:
            eta_obs = 1
            while (eta_obs < 0) or (eta_obs > 0.25):
                eta_obs = eta + sigma_eta*np.random.randn()
            theta_obs = 2
            while (theta_obs < 0) or (theta_obs > 1):
                theta_obs = theta + sigma_theta*np.random.randn()
            
        return mc_obs, sigma_mc , eta_obs,sigma_eta , rho_obs, sigma_rho, theta_obs, sigma_theta #((mc_obs, eta_obs, rho_obs, theta_obs), (sigma_mc, sigma_eta, sigma_rho, st))
    
    
    
    
    def _generate_injection_chunk(self,  N,  seed=1312, ):
        
        print('Generating events in source frame and redshift...')
        m1s, m2s, zs, thetas = self._generate_mergers(N)
        print('Generated %s events .' %str(m1s.shape))
        
        #keep = np.random.rand(N)<duty_cycle
            
        #m1s, m2s, zs, thetas = m1sGen[keep], m2sGen[keep], zsGen[keep], thetasGen[keep]
        
        #print('%s events kept assuming %s duty cycle' %(m1s.shape[0], duty_cycle))
        
        
        ## Get quantities in detector frame
        
        dLs = self.allPops.cosmo.dLGW(zs, self.H0base, self.Om0Base, self.w0Base, self.Xi0Base, self.nBase)
        m1d, m2d, = m1s*(1+zs), m2s*(1+zs)
        
        ## Get SNR
        oSNRs = self.osnr.get_oSNR(m1d, m2d, dLs)
        SNR = oSNRs*thetas
 
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
        p_draw=np.exp(log_p_draw)
        # Select
        keep = rho_obs>self.rho_th
        
        
        return m1d[keep], m2d[keep], dLs[keep], p_draw[keep]
        
        
    def generate_injections(self, N_goal, chunk_size=int(1e05), seed=1312):
        N_gen =0
        Nsucc=0
        m1s_det = []
        m2s_det = []
        dls_det = []
        wts_det = []

        enough=False
        while not enough:

            m1d, m2d, dls, wts = self._generate_injection_chunk( chunk_size, seed=seed)
    
            N_gen += chunk_size
    
            m1s_det.append(m1d)
            m2s_det.append(m2d)
            dls_det.append(dls)
            
            wts_det.append(wts)
         
            Nsucc += len(m1d)
            print('Total kept: so far= %s' %Nsucc)
        
            if Nsucc > N_goal:
                enough=True
            
        m1s_det = np.concatenate(m1s_det)
        m2s_det = np.concatenate(m2s_det)
        dls_det = np.concatenate(dls_det)
        wts_det = np.concatenate(wts_det)
        
        print('Total generated: %s' %N_gen)
        print('Total kept: %s' %Nsucc)
        
        with h5py.File(os.path.join(self.out_dir,'selected.h5'), 'w') as f:
            f.attrs['N_gen'] = N_gen
            f.create_dataset('m1det', data=m1s_det, compression='gzip', shuffle=True)
            f.create_dataset('m2det', data=m2s_det, compression='gzip', shuffle=True)
            f.create_dataset('dl', data=dls_det, compression='gzip', shuffle=True)
            f.create_dataset('wt', data=wts_det, compression='gzip', shuffle=True)
        
    
    
    def generate_dataset(self,  duty_cycle, tot_time_yrs = 5., chunks = None, seed=1312, 
                         save=False, return_vals=True, return_generated=False):
        
        '''
        Steps is the intermediate obs steps in units of yrs.
        Should bein increasing order!
        '''
        
        
        if chunks is None:
            chunks=[tot_time_yrs,]
        
        assert tot_time_yrs>=max(chunks)
        chunks.append(tot_time_yrs)
        chunks.sort()
        
        random.seed(seed)
        
  
        print('Expected events per year: %s' %self.Nperyear_expected)
        
        self.allm1s, self.allm2s, self.allzs, self.allthetas = np.array([]),  np.array([]),  np.array([]),  np.array([])
        self.allmc_obs, self.alleta_obs, self.allrho_obs, self.alltheta_obs =  np.array([]),  np.array([]),  np.array([]),  np.array([])
        self.allsigma_mc, self.allsigma_eta, self.allsigma_rho, self.allsigma_theta =  np.array([]),  np.array([]),  np.array([]),  np.array([])
        if return_generated:
            self.allm1Gen, self.allm2Gen, self.allzGen = np.array([]), np.array([]), np.array([])
        
        allNexp = []
        allDets = []
        prev_chunk = 0 #chunks[0]
        prev_ndet = 0
        allTime=0
        for chunk in chunks:
            
            ObsTimeChunk = chunk-prev_chunk
            if ObsTimeChunk==0:
                break
            
            allTime+=ObsTimeChunk
            Nexp_ = np.random.poisson(self.Nperyear_expected*(ObsTimeChunk)) # One month
            allNexp.append(Nexp_)
            prev_chunk = chunk
            
            print('\nEvents in %s years of time: %s' %(ObsTimeChunk, Nexp_))
            
            print('Generating events in source frame and redshift...')
            m1sGen, m2sGen, zsGen, thetasGen = self._generate_mergers(Nexp_)
            if return_generated:
                self.allm1Gen = np.append(self.allm1Gen, m1sGen)
                self.allm2Gen = np.append(self.allm2Gen, m2sGen)
                self.allzGen = np.append(self.allzGen, zsGen)
                
            
            print('Generated %s events .' %str(m1sGen.shape))
            
            keep = np.random.rand(Nexp_)<duty_cycle
            
            m1s, m2s, zs, thetas = m1sGen[keep], m2sGen[keep], zsGen[keep], thetasGen[keep]
        
            print('%s events kept assuming %s duty cycle' %(m1s.shape[0], duty_cycle))
            
            print('Generating observations...')
            mc_obs, sigma_mc , eta_obs,sigma_eta , rho_obs, sigma_rho, theta_obs, sigma_theta = self._generate_observations(m1s, m2s, zs, thetas )
        
            above_det_threshold = (rho_obs>self.rho_th)
            Ndet_ = above_det_threshold.sum()
            allDets.append(Ndet_)
            print('Generated %s observations .' %str(eta_obs.shape))
            print('%s events pass detection threshold of rho>%s in a period of %s yrs' %(Ndet_, self.rho_th, ObsTimeChunk))
            #print('SNRs:')
            #print(rho_obs[above_det_threshold])
            #print(sigma_mc)
            #print(sigma_eta)
            #print(sigma_rho)
            #print(sigma_theta)
            
            if Ndet_>0:
                if Ndet_ == 1 :
                    above_det_threshold = np.asarray(above_det_threshold)
                
                #print(mc_obs.shape)
                #print(mc_obs[above_det_threshold])
                #print(eta_obs[above_det_threshold])
                #print(rho_obs[above_det_threshold])
                #print(theta_obs[above_det_threshold])
                
                mc_obs_keep, eta_obs_keep, rho_obs_keep, theta_obs_keep,  = mc_obs[above_det_threshold], eta_obs[above_det_threshold], rho_obs[above_det_threshold], theta_obs[above_det_threshold]
                sigma_mc_keep, sigma_eta_keep, sigma_rho_keep, sigma_theta_keep =  sigma_mc[above_det_threshold], sigma_eta[above_det_threshold], sigma_rho[above_det_threshold], sigma_theta[above_det_threshold]
                m1s_keep, m2s_keep, zs_keep, thetas_keep = m1s[above_det_threshold], m2s[above_det_threshold], zs[above_det_threshold], thetas[above_det_threshold]
                
                #print('Primary masses :')
                #print(m1s_keep)
                #print('Secondary masses :')
                #print(m2s_keep)
                
                assert Ndet_ == len(m1s_keep)
            
                self.allm1s = np.append(self.allm1s, m1s_keep)
                self.allm2s = np.append(self.allm2s, m2s_keep)
                self.allzs = np.append(self.allzs, zs_keep)
                self.allthetas = np.append(self.allthetas, thetas_keep)
            
                self.allmc_obs = np.append(self.allmc_obs, mc_obs_keep)
                self.alleta_obs = np.append(self.alleta_obs, eta_obs_keep)
                self.allrho_obs = np.append(self.allrho_obs, rho_obs_keep)
                self.alltheta_obs = np.append(self.alltheta_obs, theta_obs_keep)
                
                #print('all SNRs so far:')
                #print(allrho_obs)
            
                self.allsigma_mc = np.append(self.allsigma_mc, sigma_mc_keep)
                self.allsigma_eta = np.append(self.allsigma_eta, sigma_eta_keep)
                self.allsigma_rho = np.append(self.allsigma_rho, sigma_rho_keep)
                self.allsigma_theta = np.append(self.allsigma_theta, sigma_theta_keep)
                
                #assert (prev_ndet+Ndet_) == len(allm1s)
                
            else:
                print(len(m1s))
                #assert Ndet_ == len(allm1s)
            
            assert (sum(allDets)) == len(self.allm1s)
             
            prev_ndet = Ndet_
            
            print('Total events detected so far: %s' %len(self.allm1s))
        
        print("allTime: %s" %allTime)
        print("tot_time_yrs: %s" %tot_time_yrs)
        assert allTime==tot_time_yrs
        
      
        if save:
            print('Saving to %s '%os.path.join(self.out_dir, 'observations.h5'))
            with h5py.File(os.path.join(self.out_dir, 'observations.h5'), 'w') as out:
                for i, Ndet_ in enumerate(allDets):
                    out.attrs['%syr'%chunks[i]] = (Ndet_, duty_cycle/chunks[i])
                
            #out.attrs['1month'] = (N1month_tot, duty_cycle/12)
            #out.attrs['1yr'] = (N1yr_tot, duty_cycle)
            #out.attrs['5yr'] = (N5yr_tot, duty_cycle*5)
                out.attrs['Tobs'] = duty_cycle*tot_time_yrs
                out.attrs['snr_th'] = self.rho_th
                print('Attributes ok.')
                def cd(n, d):
                    d = np.array(d)
                    out.create_dataset(n, data=d, compression='gzip', shuffle=True)
    
                cd('m1s', self.allm1s)
                cd('m2s', self.allm2s)
                cd('zs', self.allzs)
                cd('thetas', self.allthetas)
            
                cd('mcobs', self.allmc_obs)
                cd('etaobs', self.alleta_obs)
                cd('rhoobs', self.allrho_obs)
                cd('thetaobs', self.alltheta_obs)
                
                cd('sigma_mc', self.allsigma_mc)
                cd('sigma_eta', self.allsigma_eta)
                cd('sigma_rho', self.allsigma_rho)
                cd('sigma_t', self.allsigma_theta)
                
        if return_vals and not return_generated:
            return self.allm1s, self.allm2s, self.allzs, self.allthetas, self.allmc_obs, self.alleta_obs, self.allrho_obs, self.alltheta_obs, self.allsigma_mc, self.allsigma_eta, self.allsigma_rho, self.allsigma_theta
        elif return_vals and return_generated:
            return self.allm1s, self.allm2s, self.allzs, self.allthetas, self.allmc_obs, self.alleta_obs, self.allrho_obs, self.alltheta_obs, self.allsigma_mc, self.allsigma_eta, self.allsigma_rho, self.allsigma_theta, self.allm1Gen, self.allm2Gen, self.allzGen
    
    def _mcetathetarho_to_m1m2thetadl(self, mc, eta, theta, rho):
        
        disc = 1.0 - 4.0*eta
        m1 = 0.5*(1 + np.sqrt(disc))*mc/eta**(3.0/5.0)
        m2 = 0.5*(1 - np.sqrt(disc))*mc/eta**(3.0/5.0)
        
        
        dl = self.osnr.get_oSNR(m1, m2, np.ones(m1.shape))*theta/rho
    
        mwt = (m1+m2)**2 / (eta**(3.0/5.0)*(m1-m2))
        dlwt = dl/rho #self.osnr.get_oSNR(m1, m2, np.ones(m1.shape))*theta/(rho*rho)
    
        wt = mwt*dlwt*self.theta_p(theta)
        wt /= np.max(wt)
        r = np.random.rand(len(mc))
        s = r < wt
    
        return m1[s], m2[s], theta[s], dl[s]
    
    
    def _draw_mc(self, mc_obs, sigma_obs, size=10000):
        return np.exp(np.log(mc_obs) + sigma_obs*np.random.randn(size))

    def _draw_eta(self,eta_obs, sigma_eta, size=10000):
        ets = np.linspace(0, 0.25, 1000)
        pe = ss.norm(ets, sigma_eta)
        pets = pe.pdf(eta_obs) / (pe.cdf(0.25) - pe.cdf(0.0))
        cets = cumtrapz(pets, ets, initial=0)
        cets /= cets[-1]
        icdf = interp1d(cets, ets)
    
        return icdf(np.random.rand(size))

    def _draw_theta(self,theta_obs, sigma_theta, size=10000):
        ths = np.linspace(0, 1, 1000)
        pt = ss.norm(ths, sigma_theta)
        pths = pt.pdf(theta_obs) / (pt.cdf(1) - pt.cdf(0))
        cths = cumtrapz(pths, ths, initial=0)
        cths /= cths[-1]
        icdf = interp1d(cths, ths)
    
        return icdf(np.random.rand(size))

    def _draw_rho(self, rho_obs, size=10000):
        return rho_obs + np.random.randn(size) # Hope nothing negative!
    
    
    def get_likelihood_samples(self, Nsamples, save=True, nparallel=2, return_vals=False):
        
        #import multiprocessing as multi
        #pool = multi.Pool(processes=nparallel)
        #try:
        res =  list(map(lambda i: self._get_likelihood_samples(i, Nsamples), range(len(self.allmc_obs)))) #list(pool.imap(lambda i: self._get_likelihood_samples(i, Nsamples), range(len(self.allmc_obs))), total=len(self.allmc_obs))
        
        m1post, m2post, thetapost, dlpost = zip(*res)
        #finally:
        #    pool.close()
        #m1post, m2post, thetapost, dlpost = list(posteriors)#zip(*posteriors)
        
        
        
        if save:
            print('Saving to %s '%os.path.join(self.out_dir, 'observations.h5'))
            with h5py.File(os.path.join(self.out_dir, 'observations.h5'), 'a') as f:
                try:
                    del f['posteriors']
                except:
                    pass # Maybe it doesn't exist yet?
                pg = f.create_group('posteriors')
    
                pg.create_dataset('m1det', data=np.array(m1post), compression='gzip', shuffle=True, chunks=(1, Nsamples))
                pg.create_dataset('m2det', data=np.array(m2post), compression='gzip', shuffle=True, chunks=(1, Nsamples))
                pg.create_dataset('theta', data=np.array(thetapost), compression='gzip', shuffle=True, chunks=(1, Nsamples))
                pg.create_dataset('dl', data=np.array(dlpost), compression='gzip', shuffle=True, chunks=(1, Nsamples))
                
        if return_vals:
            return np.array(m1post), np.array(m2post), np.array(thetapost), np.array(dlpost)
    
    
    def _get_likelihood_samples(self, i, Nsamples):
        
        #print('Getting samples from the likelihood...')
        
        m1 = np.empty((0,))
        m2 = np.empty((0,))
        theta = np.empty((0,))
        dl = np.empty((0,))

        while len(m1) < Nsamples:
            a, b, c, d = self._mcetathetarho_to_m1m2thetadl(self._draw_mc(self.allmc_obs[i], self.allsigma_mc[i]),
                                              self._draw_eta(self.alleta_obs[i], self.allsigma_eta[i]),
                                              self._draw_theta(self.alltheta_obs[i], self.allsigma_theta[i]),
                                              self._draw_rho(self.allrho_obs[i]))
    
            m1 = np.concatenate((m1, a))
            m2 = np.concatenate((m2, b))
            theta = np.concatenate((theta, c))
            dl = np.concatenate((dl, d))
        
        m1post = m1[:Nsamples]
        m2post = m2[:Nsamples]
        thetapost = theta[:Nsamples]
        dlpost = dl[:Nsamples]
        
        return m1post, m2post, thetapost, dlpost
            
            