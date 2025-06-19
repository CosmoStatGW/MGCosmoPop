#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 12:12:51 2022

@author: Michi
"""


import numpy as np
import random
import h5py
import os 
import sys
from tqdm import tqdm

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))


from scipy.interpolate import interp1d

try:
    from scipy.integrate import cumtrapz as mycumtrapz
except:
    from scipy.integrate import cumulative_trapezoid as mycumtrapz

import scipy.stats as ss

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
                 zmax, 
                 out_dir,
                 snr_th_dets = [],
                 add_noise=False,
                eta_scatter=5e-03, mc_scatter = 3e-02, theta_scatter = 5e-02,
                 seed=None,
                 condition='and'
#                 **kwargs):
    ):        


        self.condition=condition
        self.add_noise = add_noise
        self.seed=seed
        
        self.out_dir=out_dir
        self.detNet = detNet 
        
        self.allPops=populations
        
        self.zmax = zmax 
        
        #self.snr_th=snr_th
        self.snr_th_dets=snr_th_dets
        self.snr_th_net=snr_th_dets['net']

        self.eta_scatter=eta_scatter
        self.mc_scatter=mc_scatter
        self.theta_scatter=theta_scatter
        
    
        
        self.lambdaBase = self.allPops.get_base_values(self.allPops.params)
        self.LambdaCosmoBase, self.LambdaAllPopBase = self.allPops._split_params(self.lambdaBase)
        self.lambdaBBHrate, self.lambdaBBHmass, self.lambdaBBHspin = self.allPops._pops[0]._split_lambdas(self.LambdaAllPopBase)
        
        self.H0base, self.Om0Base, self.w0Base, self.Xi0Base, self.nBase = self.allPops.cosmo._get_values(self.LambdaCosmoBase, [ 'H0', 'Om', 'w0', 'Xi0', 'n'])
    
        self._find_Nperyear_expected()
        self._get_theta_pdf()
    
    def _find_Nperyear_expected(self):  
         self.Nperyear_expected = self.allPops.Nperyear_expected(self.lambdaBase, zmax=self.zmax, verbose=True)
         print('Expected number per year between redshift 0 and %s: %s'%(self.zmax, self.Nperyear_expected) )

    def _get_theta_pdf(self, N=100000):
        try:
            thpath = os.path.join(Globals.thetaPath, 'thetas.h5')
            
            with h5py.File(thpath, 'r') as f:
                ts = np.array(f['Theta'])
        except:
    
            cos_th = np.random.uniform(low=-1, high=1, size=N)
            theta = np.arccos(cos_th)
            cos_iota = np.random.uniform(low=-1, high=1, size=N)
            phi = np.random.uniform(low=0, high=2*np.pi, size=N)
            psi = np.random.uniform(low=0, high=2*np.pi, size=N)

            d = list(self.detNet.signals.keys())[0]
            Fp, Fc = self.detNet.signals[d]._PatternFunction(theta, phi, t=np.zeros(N), psi=psi, rot=0. )

            ts = np.sqrt( 0.25*Fp**2*(1 + cos_iota**2)**2 + Fc**2*cos_iota**2 )

        ts = np.concatenate((ts, [0.0, 1.0]))
        ts = np.sort(ts)  
        kde = ss.gaussian_kde(ts)
        tts = np.linspace(0, 1, 1000)
        ps = kde(tts) + kde(-tts) + kde(2-tts)
        
        self.theta_p = interp1d(tts, ps)
        self.theta_cdf =  interp1d( ts, np.linspace(0, 1, ts.shape[0]))

    #################################################################################
    #################################################################################
    # Generate events from the population
    #################################################################################
    #################################################################################
    def _generate_mergers(self, N, verbose=True):
        
        #np.random.seed(self.seed)
        #np.random.seed(np.random.randint(2**32 - 1, size=1))
        
        if N==1:
            verbose=False
        if verbose:
            print('Parameters used to sample events: %s' %str(self.lambdaBase))
        
        massSamples, zs, spinSamples = self.allPops.sample(N, self.zmax, self.lambdaBase)
        
        m1s, m2s = np.squeeze(massSamples)[:, 0], np.squeeze(massSamples)[:, 1]

        if self.allPops._pops[0].spinDist.__class__.__name__ =='UniformSpinDistChiz':
        
            spin1z, spin2z = np.squeeze(spinSamples)[:, 0], np.squeeze(spinSamples)[:, 1]
            spins='flat'
        elif self.allPops._pops[0].spinDist.__class__.__name__ =='DefaultSpinModel':
            spins='default'
            chi1, chi2, cost1, cost2 = np.squeeze(spinSamples)[:, 0], np.squeeze(spinSamples)[:, 1], np.squeeze(spinSamples)[:, 2], np.squeeze(spinSamples)[:, 3]
            spin1z = chi1*cost1
            spin2z = chi2*cost2
        elif self.allPops._pops[0].spinDist.__class__.__name__ =='UniformOnSphereSpin':
            spins='uniform_on_sphere'
            s1x, s1y, s1z, s2x, s2y, s2z = np.squeeze(spinSamples)[:, 0], np.squeeze(spinSamples)[:, 1], np.squeeze(spinSamples)[:, 2], np.squeeze(spinSamples)[:, 3], np.squeeze(spinSamples)[:, 4], np.squeeze(spinSamples)[:, 5]
            spin1z = s1z
            spin2z = s2z
        elif self.allPops._pops[0].spinDist.__class__.__name__ =='DummySpinDist':
            spin1z = np.zeros(m1s.shape)
            spin2z = np.zeros(m1s.shape)
            spins='skip'
        else:
            raise ValueError()
        
        costheta = np.random.uniform(-1, 1, N)
        cosiota = np.random.uniform(-1, 1, N)
            
        theta = np.arccos(costheta)
        iota = np.arccos(cosiota)
            
        phi = np.random.uniform(0, np.pi*2, N)
        phiCoal = np.random.uniform(0, np.pi*2, N)
        psi = np.random.uniform(0, np.pi, N)
            
        tcoal = np.random.uniform(size=N)
        
        Lambda1 = np.random.uniform(low=0, high=2000, size=N)
        Lambda2 = np.random.uniform(low=0, high=2000, size=N)
        
        ## Get quantities in detector frame
        
        dL = self.allPops.cosmo.dLGW(zs, self.H0base, self.Om0Base, self.w0Base, self.Xi0Base, self.nBase)
        mass1_det, mass2_det, = m1s*(1+zs), m2s*(1+zs)
        
        Mc_det = (mass1_det*mass2_det)**(3/5)/((mass1_det+mass2_det)**(1/5))
        eta = (mass1_det*mass2_det)/((mass1_det+mass2_det)**(2))
        
        
        events = { 'Mc':Mc_det,
          'eta':eta,
          'm1_source':m1s,
          'm2_source':m2s,
          'm1_det':mass1_det,
          'm2_det':mass2_det,
          'z':zs,
          'dL':dL,
          'theta':theta, 
          'phi':phi,
          
          'ra':phi,
          'dec':np.pi/2-theta,
          
          'iota':iota,
          'psi':psi, 
                  
          #'tGPS':gpsts,
          'Phicoal':phiCoal ,
                  
          'Lambda1': Lambda1,
          'Lambda2':  Lambda2,
                
                     #'chi1': chi1,
                     #'chi2':chi2,
                    #'cost1': cost1,
                    #'cost2':cost2,
            'chi1z': spin1z,
            'chi2z':spin2z,
          'tcoal':tcoal    
}

        if spins=='default':
            events['chi1'] = chi1
            events['chi2'] = chi2
            events['cost1'] = cost1
            events['cost2'] = cost2
        elif spins=='uniform_on_sphere':
            events['s1x'] = s1x
            events['s2x'] = s2x
            events['s1y'] = s1y
            events['s2y'] = s2y
            events['s1z'] = s1z
            events['s2z'] = s2z
        
        return events, spins #np.squeeze(m1s), np.squeeze(m2s), np.squeeze(zs), costhetas,  phis, cosiotas, ts_det
    
    

    #################################################################################
    #################################################################################
    # Generate injections
    #################################################################################
    #################################################################################
    
    def _generate_injection_chunk(self,  N, verbose=False ):
        
        # necessary !! otherwise get repeated events
        np.random.seed()
        
        #if verbose:
        #    print('Generating events in source frame and redshift...')
        events_injected, spins = self._generate_mergers(N, verbose=False)
        if verbose:
            print('Generated %s events .' %str(events_injected['dL'].shape))
      
        
        ## Get SNR
        if verbose:
            print('Computing snr...')
        try:
            allSNRsOpt = self.detNet.SNR_netFast(events_injected, return_all=True)
        except:
            allSNRsOpt = self.detNet.SNR(events_injected, return_all=True)
        
        SNRopt =  allSNRsOpt['net'] # network snr
        if verbose:
            print('Done.')
        
        
        if self.add_noise:
            if verbose:
                print('Adding noise...')
            # Get observed SNR (ie gaussian random variable w unit variance for each det). 
            SNRobsSq=np.zeros(allSNRsOpt['net'].shape)
            allSNRs = {}
            for d in allSNRsOpt.keys():
                if d!='net':
                    if verbose:
                        print('Adding %s' %d)
                    SNRobs_ = allSNRsOpt[d]+np.random.normal(loc=0, scale=1, size=len(allSNRsOpt[d]))
                    allSNRs[d] = SNRobs_
                    SNRobsSq += allSNRs[d]**2
            SNR = np.sqrt(SNRobsSq)
            allSNRs['net'] = SNR
        else:
            # Use optimal SNR. 
            if verbose:
                print('Using optimal SNR...')
            SNR = SNRopt
            allSNRs = allSNRsOpt
     
        
        # Select
        if self.condition=='and':       
            keep = np.full(SNR.shape, True)
        elif self.condition=='or':
            keep = np.full(SNR.shape, False)

        m_ = allSNRs['net']>8
        print('%s events with net snr larger than 8:'%m_.sum())
        
        print({k:allSNRs[k][m_] for k in allSNRs.keys()})
        
        for d in allSNRs.keys():
            if verbose:
                print('Searching %s in %s' %(d , str(list(self.snr_th_dets.keys()))))
            found=False
            for d1 in self.snr_th_dets.keys():
                if d1 in d and not found:
                    th = self.snr_th_dets[d1]
                    if verbose:
                        print('th for %s: %s' %(d, th))
                    found=True
            if found:
                if self.condition=='and':           
                    keep &= allSNRs[d]>th
                elif self.condition=='or':
                    keep |= allSNRs[d]>th
                if verbose:
                    print('Kept after %s snr cut: %s' %(d, keep.sum()))
        ndet = keep.sum()
        
        events_detected =  {k: events_injected[k][keep] for k in events_injected.keys()}
        
        # if not including spins in the population model, and using flat distribution for injections, 
        # here do not pass the spins. this way the log prob will be already correct
        # otherwise, pass spins here and include them in the pop model.
        
        if ndet>0:

            
            log_p_draw_nospin = self.allPops.log_dN_dm1zdm2zddL( events_detected['m1_source'], events_detected['m2_source'], events_detected['z'], 
                                                     #[events_detected['chi1z'], events_detected['chi2z']], 
                                                      [],
                                                     self.lambdaBase, 1., dL=events_detected['dL'])-np.log(self.Nperyear_expected)

            if spins=='default':
                log_p_draw_spin = self.allPops._pops[0].spinDist.logpdf([events_detected['chi1'], events_detected['chi2'], events_detected['cost1'], events_detected['cost2']], self.lambdaBBHspin)
            elif spins=='flat':
                log_p_draw_spin = self.allPops._pops[0].spinDist.logpdf([events_detected['chi1z'], events_detected['chi2z']], self.lambdaBBHspin)
            elif spins=='uniform_on_sphere':
                log_p_draw_spin = self.allPops._pops[0].spinDist.logpdf([events_detected['s1x'], events_detected['s1y'], events_detected['s1z'], events_detected['s2x'], events_detected['s2y'], events_detected['s2z']], self.lambdaBBHspin)
            else:
                log_p_draw_spin =  np.ones(events_detected['m1_source'].shape)

            log_p_draw =  log_p_draw_spin + log_p_draw_nospin
    
        
        else:
            log_p_draw=np.full( events_detected['m1_source'].shape, np.inf)
            log_p_draw_nospin=np.full( events_detected['m1_source'].shape, np.inf)
        
        events_detected['log_p_draw'] = log_p_draw
        events_detected['log_p_draw_nospin'] = log_p_draw_nospin
        
        for k in allSNRs.keys():
            events_detected['snr_%s'%k] = allSNRs[k][keep]
        
        events_detected['snr'] = SNR[keep]
        
        
        return events_detected
        
        
    def generate_injections(self, N_goal, chunk_size=int(1e05),):
        N_gen =0
        Nsucc=0
        m1s_det = []
        m2s_det = []
        dls_det = []
        logwts_det = []
        logwts_nosp_det = []
        enough=False
        #first_it=True
        success=False
        ntry=2
        maxit=3
        it=0
        all_evs_det = {}
        while not success:
            try:
                tmp_ = self._generate_injection_chunk( ntry, verbose=True)
                success=True
            except Exception as e:
                print(e)
                ntry+=2
                it+=1
                if it>=maxit:
                    #success=True
                    ntry=chunk_size
                #tmp_ = self._generate_injection_chunk( ntry, verbose=False)
        
        rhos_det_all = {k:[] for k in tmp_.keys() if 'snr' in k }
        print('\nStart...\n')
        is_first=True
        while not enough:

            #m1d, m2d, dls, logwts, rho = self._generate_injection_chunk( chunk_size, )
            events_chunk = self._generate_injection_chunk( chunk_size, verbose=True )
            
            if is_first:
                all_evs_det = events_chunk
                is_first=False
            else:
                for k in all_evs_det.keys():
                    all_evs_det[k] = np.append( all_evs_det[k], events_chunk[k])
                    if k=='dL':
                        print('Events detected cat len: %s' %len(all_evs_det[k]))
            
            m1d, m2d, dls, logwts, logwts_nosp = np.squeeze(events_chunk['m1_det']), np.squeeze(events_chunk['m2_det']), np.squeeze(events_chunk['dL']), np.squeeze(events_chunk['log_p_draw']), np.squeeze(events_chunk['log_p_draw_nospin'])
            
            
            try:
                Nsucc += len(m1d)
            except Exception as e:
                #print(e)
                if np.ndim(m1d)==0 or np.isscalar(m1d):
                    Nsucc += 1
                pass
            
            if np.ndim(m1d)==0 or np.isscalar(m1d):
                m1d, m2d, dls, logwts, logwts_nosp = np.array([m1d]), np.array([m2d]), np.array([dls]), np.array([logwts]), np.array([logwts_nosp])
            
            N_gen += chunk_size
            
            #if first_it:
            #   print('Fisrt iteration !')
            #    for k in events_chunk.keys():
            #        if 'snr' in k:
            #            rhos_det_all[k] = events_chunk[k]
            #    first_it=False
            #else:
            for k in rhos_det_all.keys():
                    rhos_det_all[k].append(events_chunk[k])
                #first_it=False

            
            m1s_det.append(m1d)
            m2s_det.append(m2d)
            dls_det.append(dls)
            #rhos_det.append(rho)
            logwts_det.append(logwts)
            logwts_nosp_det.append(logwts_nosp)

            
            print('\nTotal kept: so far= %s\n\n' %Nsucc)
        
            if Nsucc > N_goal:
                enough=True
            
        m1s_det = np.concatenate(m1s_det)
        m2s_det = np.concatenate(m2s_det)
        dls_det = np.concatenate(dls_det)
        logwts_det = np.concatenate(logwts_det)
        logwts_nosp_det = np.concatenate(logwts_nosp_det)
        #rhos_det =  np.concatenate(rhos_det)
        for k in rhos_det_all.keys():
            rhos_det_all[k] = np.concatenate(rhos_det_all[k])
        print('Total generated: %s' %N_gen)
        print('Total kept: %s' %Nsucc)
        
        
        
        
        print('Saving catalog of detected events...')
        with h5py.File(os.path.join(self.out_dir,'catalog_detected.h5'), 'w') as out:
                        
            def cd(n, d):
                d = np.array(d)
                out.create_dataset(n, data=d, compression='gzip', shuffle=True)
            
            for key in all_evs_det.keys():
                cd(key, all_evs_det[key])
        print('Done.')
        print('Saving in MGCosmoPop format...')
        with h5py.File(os.path.join(self.out_dir, 'selected.h5'), 'w') as f:
            f.attrs['N_gen'] = N_gen
            for k in self.snr_th_dets.keys():
                f.attrs['snr_th_%s'%k] = self.snr_th_dets[k]
                if k=='net':
                    f.attrs['snr_th'] = self.snr_th_dets[k]
            f.create_dataset('m1det', data=m1s_det, compression='gzip', shuffle=True)
            f.create_dataset('m2det', data=m2s_det, compression='gzip', shuffle=True)
            f.create_dataset('dl', data=dls_det, compression='gzip', shuffle=True)
            f.create_dataset('logwt', data=logwts_det, compression='gzip', shuffle=True)
            f.create_dataset('logwt_nosp', data=logwts_nosp_det, compression='gzip', shuffle=True)
            #f.create_dataset('snr', data=rhos_det, compression='gzip', shuffle=True)
            for k in rhos_det_all.keys():
                print('Saving snr with key %s' %k)
                f.create_dataset(k, data=rhos_det_all[k], compression='gzip', shuffle=True)
        print('Done.')
        print('Saving in pymcpop format...')
        np.save( os.path.join(self.out_dir+'_dL.npy'), dls_det )
        np.save( os.path.join(self.out_dir+'_m1d.npy'), m1s_det )
        np.save( os.path.join(self.out_dir+'_m2d.npy'), m2s_det )
        
        np.save( os.path.join(self.out_dir+'_log_p_draw.npy'), logwts_nosp_det )
        np.save( os.path.join(self.out_dir+'_Ngen.npy'), N_gen )
        np.savetxt( os.path.join(self.out_dir+'_Tobs.txt'), np.array([1.])) 
        np.save( os.path.join(self.out_dir+'_log_p_draw_spin.npy'), logwts_det )
        print('Done.')


    #################################################################################
    #################################################################################
    # Generate posterior samples
    #################################################################################
    #################################################################################
    
    def generate_dataset(self,  duty_cycle, tot_time_yrs = 5., chunks = None, seed=1312, 
                         save=False, return_vals=True, return_generated=False):

        # Input : list of *true* source-frame masses, redshift, ... of the generated population
        # Output: *observed* quantities after snr threshold and adding noise 
        # Also saves to observations.h5

        # chunks is the intermediate obs steps in units of yrs.
        # Should be in increasing order!


        if chunks is None:
            chunks = [tot_time_yrs, ]
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
            Nexp_ = np.random.poisson( self.Nperyear_expected*(ObsTimeChunk))
            allNexp.append(Nexp_)
            prev_chunk = chunk
            
            print('\nEvents in %s years of time: %s' %(ObsTimeChunk, Nexp_))
            
            print('Generating events in source frame and redshift...')
            events, spins = self._generate_mergers(Nexp_)
            
            m1sGen = events['m1_source']
            m2sGen = events['m2_source']
            zsGen = events['z']
            cos_iota = np.cos(events['iota'])

            # compute Theta from Finn-Chernoff
            # assume single detector in the net
            d = list(self.detNet.signals.keys())[0]
            Fp, Fc = self.detNet.signals[d]._PatternFunction(events['theta'], events['phi'], t=np.zeros(m1sGen.shape), psi=events['psi'], rot=0. )

            
            thetasGen = np.sqrt( 0.25*Fp**2*(1 + cos_iota**2)**2 + Fc**2*cos_iota**2 )
            
            
            if return_generated:
                self.allm1Gen = np.append(self.allm1Gen, m1sGen)
                self.allm2Gen = np.append(self.allm2Gen, m2sGen)
                self.allzGen = np.append(self.allzGen, zsGen)
                
            
            print('Generated %s events .' %str(m1sGen.shape))
            
            keep = np.random.rand(Nexp_)<duty_cycle
            
            m1s, m2s, zs, thetas = m1sGen[keep], m2sGen[keep], zsGen[keep], thetasGen[keep]
        
            print('%s events kept assuming %s duty cycle' %(m1s.shape[0], duty_cycle))
            
            print('Generating observations...')
            mc_obs_keep, sigma_mc_keep , eta_obs_keep,sigma_eta_keep , rho_obs_keep, sigma_rho_keep, theta_obs_keep, sigma_theta_keep, Ndet_, above_det_threshold = self._generate_observations(m1s, m2s, zs, thetas )

            allDets.append(Ndet_)
            print('%s events pass detection threshold of rho_obs>%s in a period of %s yrs' %(Ndet_, self.snr_th_net, ObsTimeChunk))
          
            
            if Ndet_>0:
                m1s_keep, m2s_keep, zs_keep, thetas_keep = m1s[above_det_threshold], m2s[above_det_threshold], zs[above_det_threshold], thetas[above_det_threshold]
                               
                assert Ndet_ == len(m1s_keep)
            
                self.allm1s = np.append(self.allm1s, m1s_keep)
                self.allm2s = np.append(self.allm2s, m2s_keep)
                self.allzs = np.append(self.allzs, zs_keep)
                self.allthetas = np.append(self.allthetas, thetas_keep)
            
                self.allmc_obs = np.append(self.allmc_obs, mc_obs_keep)
                self.alleta_obs = np.append(self.alleta_obs, eta_obs_keep)
                self.allrho_obs = np.append(self.allrho_obs, rho_obs_keep)
                self.alltheta_obs = np.append(self.alltheta_obs, theta_obs_keep)
                
                self.allsigma_mc = np.append(self.allsigma_mc, sigma_mc_keep)
                self.allsigma_eta = np.append(self.allsigma_eta, sigma_eta_keep)
                self.allsigma_rho = np.append(self.allsigma_rho, sigma_rho_keep)
                self.allsigma_theta = np.append(self.allsigma_theta, sigma_theta_keep)
                                
            else:
                print(len(m1s))
            
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
                

                out.attrs['Tobs'] = duty_cycle*tot_time_yrs
                out.attrs['snr_th'] = self.snr_th_net
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




    def get_likelihood_samples(self, Nsamples, save=True, nparallel=2, return_vals=False, out_path=None, seed=1312, samples_step=100):
        # Generate posterior samples from the likelihood model for events in the catalog
        # saves to observations.h5 - field 'posteriors'

        random.seed(seed)
        
        #import multiprocessing as multi
        #pool = multi.Pool(processes=nparallel)
        #try:
        #res =  list(map(lambda i: self._get_likelihood_samples(i, Nsamples,), range(len(self.allmc_obs)) ))
        res = list(map(
                lambda i: self._get_likelihood_samples(i, Nsamples, size=samples_step),
                    tqdm(range(len(self.allmc_obs)), desc="Sampling")
                ))
        
        #list(pool.imap(lambda i: self._get_likelihood_samples(i, Nsamples), range(len(self.allmc_obs))), total=len(self.allmc_obs))
        
        m1post, m2post, thetapost, dlpost = zip(*res)
        #finally:
        #    pool.close()
        #m1post, m2post, thetapost, dlpost = list(posteriors)#zip(*posteriors)

        print('Final samples shape: %s' %str(np.array(m1post).shape))

        
        if save:
            print('Saving to %s '%os.path.join(self.out_dir, 'observations.h5'))
            with h5py.File(os.path.join(self.out_dir, 'observations.h5'), 'a') as f:
                try:
                    del f['posteriors']
                except:
                    pass 
                pg = f.create_group('posteriors')
    
                pg.create_dataset('m1det', data=np.array(m1post), compression='gzip', shuffle=True, chunks=(1, Nsamples))
                pg.create_dataset('m2det', data=np.array(m2post), compression='gzip', shuffle=True, chunks=(1, Nsamples))
                pg.create_dataset('theta', data=np.array(thetapost), compression='gzip', shuffle=True, chunks=(1, Nsamples))
                pg.create_dataset('dl', data=np.array(dlpost), compression='gzip', shuffle=True, chunks=(1, Nsamples))


        
        if out_path is not None:
            import seaborn as sns
            import matplotlib.pyplot as plt
            sns.displot((np.mean(thetapost, axis=-1)-self.allthetas)/(np.std(thetapost, axis=-1)) )
            plt.axvline(0, ls='--', color='k');
            plt.savefig(os.path.join(out_path, 'thetas_zscore.pdf') )
            plt.close()

            allm1d = self.allm1s*(1+self.allzs)
            allm2d = self.allm2s*(1+self.allzs)
            
            sns.displot((np.mean(m1post, axis=-1)-allm1d)/(np.std(m1post, axis=-1)) )
            plt.axvline(0, ls='--', color='k');
            plt.savefig(os.path.join(out_path, 'm1s_zscore.pdf') )
            plt.close()

            sns.displot((np.mean(m2post, axis=-1)-allm2d)/(np.std(m2post, axis=-1)) )
            plt.axvline(0, ls='--', color='k');
            plt.savefig(os.path.join(out_path, 'm2s_zscore.pdf') )
            plt.close()

            self.alldLs = self.allPops.cosmo.dLGW(self.allzs, self.H0base, self.Om0Base, self.w0Base, self.Xi0Base, self.nBase)
            sns.displot((np.mean(dlpost, axis=-1)-self.alldLs)/(np.std(dlpost, axis=-1)) )
            plt.axvline(0, ls='--', color='k');
            plt.savefig(os.path.join(out_path, 'dLs_zscore.pdf') )
            plt.close()        
        

#        random.set_state(old_state)
        if return_vals:
            return np.array(m1post), np.array(m2post), np.array(thetapost), np.array(dlpost)
    
    
    def _get_likelihood_samples(self, i, Nsamples, size=100):
        if i%100==0 and i!=0:
            print('Step n. %s' %i)
        
        allm1 = np.empty((0,))
        allm2 = np.empty((0,))
        alltheta = np.empty((0,))
        alldl = np.empty((0,))

        while len(allm1) < Nsamples:

            MCdraws = self._draw_mc(self.allmc_obs[i], self.allsigma_mc[i], size=size)
            etaDraws = self._draw_eta(self.alleta_obs[i], self.allsigma_eta[i], size=size)
            disc = 1.0 - 4.0*etaDraws
            m1 = 0.5*(1 + np.sqrt(disc))*MCdraws/etaDraws**(3.0/5.0)
            m2 = 0.5*(1 - np.sqrt(disc))*MCdraws/etaDraws**(3.0/5.0)

            not_good = (m1<1) | (m2<1) | (m1>1000) #| (etaDraws<0.25)
            check=False
            if np.any(not_good):
                check=True
                print('Replacing %s bad draws for i=%s...' %(not_good.sum(), i))
                j=0
            while np.any(not_good):
                MCdraws[not_good] = self._draw_mc(self.allmc_obs[i], self.allsigma_mc[i], size=not_good.sum())
                etaDraws[not_good] = self._draw_eta(self.alleta_obs[i], self.allsigma_eta[i], size=not_good.sum())
                disc = 1.0 - 4.0*etaDraws
                m1 = 0.5*(1 + np.sqrt(disc))*MCdraws/etaDraws**(3.0/5.0)
                m2 = 0.5*(1 - np.sqrt(disc))*MCdraws/etaDraws**(3.0/5.0)
                not_good = (m1<1) | (m2<1) | (m1>1000)
                if j%100==0:
                    print('Trial N. %s, number left to replace: %s' %(j, not_good.sum()))
                j+=1
            if not np.any(not_good) and check:
                print('Success.')

            m1_, m2_, th_, d_ = self._mcetathetarho_to_m1m2thetadl(MCdraws,
                                                            etaDraws,
                                              self._draw_theta(self.alltheta_obs[i], self.allsigma_theta[i], size=size),
                                              self._draw_rho(self.allrho_obs[i], size=size))
    
            allm1 = np.concatenate((allm1, m1_))
            allm2 = np.concatenate((allm2, m2_))
            alltheta = np.concatenate((alltheta, th_))
            alldl = np.concatenate((alldl, d_))
            
            
        allm1 = np.asarray(allm1).flatten()
        allm2 = np.asarray(allm2).flatten()
        alltheta = np.asarray(alltheta).flatten()
        alldl = np.asarray(alldl).flatten()
        
        m1post = allm1[:Nsamples]
        m2post = allm2[:Nsamples]
        thetapost = alltheta[:Nsamples]
        dlpost = alldl[:Nsamples]


        
        return m1post, m2post, thetapost, dlpost


    def get_oSNR(self, m1d, m2d, dLs):

        Mc_det = (m1d*m2d)**(3/5)/((m1d+m2d)**(1/5))
        eta = (m1d*m2d)/((m1d+m2d)**(2))

        # optimal snrs. the (single) detector must be located at \lambda=\pi/2, \varphi = 0, \gamma=\pi/4
        
        events_for_osnr = events = { 'Mc':Mc_det,
          'eta':eta,
          'dL':dLs,
                                    
          'theta':np.zeros(Mc_det.shape), 
          'phi':np.zeros(Mc_det.shape),
          
          'iota':np.zeros(Mc_det.shape),
          'psi':np.zeros(Mc_det.shape), 
                  
          'Phicoal':np.zeros(Mc_det.shape) ,
            'chi1z': np.zeros(Mc_det.shape),
            'chi2z':np.zeros(Mc_det.shape),
          'tcoal':np.zeros(Mc_det.shape)    
}

        try:
            allSNRsOpt = self.detNet.SNR_netFast(events_for_osnr, return_all=True)
        except:
            allSNRsOpt = self.detNet.SNR(events_for_osnr, return_all=True)
        
        return allSNRsOpt['net'] # network snr


    def _generate_observations(self, m1s, m2s, zs, theta, verbose=True, eps=0.001):

        # called internally by generate_dataset at every loop.
        # Input : list of *true* source-frame masses, redshift, ... of the generated population
        # Output: *observed* quantities after snr threshold and adding noise 
        

        if np.isscalar(m1s) or m1s.ndim==0:
            dim = 1
            verbose=False
        else:
            dim = m1s.shape[0]
        
        ## Get quantities in detector frame
        # print('COSMO PARAMETERS FOR DL: %s' %str([self.H0base, self.Om0Base, self.w0Base, self.Xi0Base, self.nBase]))
        dLs = self.allPops.cosmo.dLGW(zs, self.H0base, self.Om0Base, self.w0Base, self.Xi0Base, self.nBase)
        m1d, m2d, = m1s*(1+zs), m2s*(1+zs)
        #print('Example: z = %s, dLs: %s' %(str(zs[:5]), str(dLs[:5])) )

        
        oSNRs = self.get_oSNR(m1d, m2d, dLs)
        
    
        SNR = oSNRs*theta
        if verbose:
            print('N. of opt SNRs >%s: %s' %(self.snr_th_net, str(oSNRs[oSNRs>self.snr_th_net].shape)))
            print('N. of raw SNRs >%s: %s' %(self.snr_th_net, str(SNR[SNR>self.snr_th_net].shape)))

        # To spped up, throw away immediately "hopeless" events with very low SNR
#        hopeless = SNR<0.01
#        m1d, m2d, dLs, zs, theta, SNR = m1d[~hopeless], m2d[~hopeless], dLs[~hopeless], zs[~hopeless], theta[~hopeless], SNR[~hopeless]
#        print('%s hopeless events with SNR<0.01 thrown away' %(hopeless.sum()))
#        dim= m1d.shape[0]
#        print('N. of remaining events: %s' %dim)
        
        
        rho_obs = SNR + np.random.randn(dim)
        out = rho_obs<0
        print('Imposing observed SNR>0...')
        while np.any(out):
            replace = SNR[out]+np.random.randn(rho_obs[out].shape[0])
            rho_obs[out] = replace
            out = rho_obs<0

        import matplotlib.pyplot as plt
        _ = plt.hist(SNR, density=False, bins=40, alpha=0.5, label='SNR theoretical')
        _ = plt.hist(rho_obs, density=False, bins=40, alpha=0.5, label='SNR obs')
        plt.legend(fontsize=18)
        plt.axvline(self.snr_th_net)
#        plt.xscale('log')
        plt.savefig(os.path.join(self.out_dir,'SNRdist.pdf'))
        plt.close()


        above_det_threshold = (rho_obs>self.snr_th_net)
        Ndet_ = above_det_threshold.sum()
        print('Generated %s observations .' %str(rho_obs.shape))

        
        if Ndet_>0:
                if Ndet_ == 1 :
                    above_det_threshold = np.asarray(above_det_threshold)

        m1d, m2d, dLs, zs, theta, SNR, rho_obs = m1d[above_det_threshold], m2d[above_det_threshold], dLs[above_det_threshold], zs[above_det_threshold], theta[above_det_threshold], SNR[above_det_threshold], rho_obs[above_det_threshold]

        dim = m1d.shape[0] 
        
        sigma_rho = np.ones(dim)
    
        mtot = m1d+m2d
        eta = m1d*m2d/(mtot*mtot)
        mc = mtot*eta**(3.0/5.0)
    
        sigma_mc = self.snr_th_net/rho_obs*self.mc_scatter
        assert np.all(sigma_mc>0)
        mc_obs = np.random.lognormal(mean=np.log(mc), sigma=sigma_mc)
    
        sigma_eta = self.snr_th_net/rho_obs*self.eta_scatter
        sigma_theta = self.snr_th_net/rho_obs*self.theta_scatter

        
        if dim>1:      
            eta_obs = eta + sigma_eta*np.random.randn(dim)
            out = (eta_obs<0) | (eta_obs>0.25)
            print('Imposing cut on eta...')
            npoints=0
            NrepPr=0
            while np.any(out):
                replace =  eta[out] + sigma_eta[out]*np.random.randn(eta[out].shape[0])
                Nrep = out.sum()
                if Nrep<NrepPr-1000:
                    print('N. of points to replace: %s' %str(Nrep))
                NrepPr=Nrep
                eta_obs[out] = replace  
                out = (eta_obs<0) | (eta_obs>0.25)
            assert np.all( (eta_obs>0) & (eta_obs<0.25))
        
            
            theta_obs = theta + sigma_theta*np.random.randn(dim)
            out = (theta_obs<0) | (theta_obs>1)
            print('Imposing cut on Theta...')
            npoints=0
            while np.any(out):
                replace=theta[out] + sigma_theta[out]*np.random.randn(theta[out].shape[0])
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
            
        return mc_obs, sigma_mc , eta_obs,sigma_eta , rho_obs, sigma_rho, theta_obs, sigma_theta, Ndet_, above_det_threshold

      



    #################################################################################
    # auxiliary stuff to sample according to the likelihood model
    def _mcetathetarho_to_m1m2thetadl(self, mc, eta, theta, rho):
        
        disc = 1.0 - 4.0*eta
        m1 = 0.5*(1 + np.sqrt(disc))*mc/eta**(3.0/5.0)
        m2 = 0.5*(1 - np.sqrt(disc))*mc/eta**(3.0/5.0)

        
        dl = self.get_oSNR(m1, m2, np.ones(m1.shape))*theta/rho

        mwt = (m1+m2)**2 / (eta**(3.0/5.0)*(m1-m2))
        dlwt = dl/rho #self.osnr.get_oSNR(m1, m2, np.ones(m1.shape))*theta/(rho*rho)
    
        wt = mwt*dlwt*self.theta_p(theta)
        wt /= np.max(wt)
        r = np.random.rand(len(mc))
        s = r < wt

        return m1[s], m2[s], theta[s], dl[s]
    
    
    def _draw_mc(self, mc_obs, sigma_obs, size=1000):
        return np.exp(np.log(mc_obs) + sigma_obs*np.random.randn(size))

        
    def _draw_eta(self,eta_obs, sigma_eta, size=1000):
        ets = np.linspace(0, 0.25, 1000)
        pe = ss.norm(ets, sigma_eta)
        pets = pe.pdf(eta_obs) / (pe.cdf(0.25) - pe.cdf(0.0))
        cets = mycumtrapz(pets, ets, initial=0)
        cets /= cets[-1]
        icdf = interp1d(cets, ets)
        return icdf(np.random.rand(size))

    def _draw_theta(self, theta_obs, sigma_theta, size=1000):
        ths = np.linspace(0, 1, 1000)
        pt = ss.norm(ths, sigma_theta)
        pths = pt.pdf(theta_obs) / (pt.cdf(1) - pt.cdf(0))
        cths = mycumtrapz(pths, ths, initial=0)
        cths /= cths[-1]
        icdf = interp1d(cths, ths)
    
        return icdf(np.random.rand(size))

    def _draw_rho(self, rho_obs, size=1000):
    
        r = rho_obs + np.random.randn(size) 
        not_good=(r<0)
        while np.any(not_good):
            r[not_good] = rho_obs + np.random.randn(r[not_good].shape[0])
            not_good = r<0 
        return r


        


