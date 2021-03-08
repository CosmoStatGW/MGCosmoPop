#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 16:19:57 2021

@author: Michi
"""
#import astropy.units as u
#import numpy as np
#import scipy.stats as ss
from ..ABSpopulation import Population
from copy import deepcopy

########################################################################
########################################################################


class AstroPopulation(Population):
     
    '''
    Base class for astrophysical models.
    Made of three components:
        - rate evolution as function of redshift dR/dVdt (In the simplest case, the differential rate is given by
                                                          dR/dVdt = R0*(1+z)**lambda)
        - mass distribution p(m1, m2)
        - spin distribution p(s1, s2)
    Each of the three is implemented in a separate class
    
    
    
    then dR/dm1dm2 = dR/dVdt * p(m1, m2)* p(s1, s2)
    
    '''

    
    
    def __init__(self, rateEvolution, massDistribution, spinDistribution ):
        Population.__init__(self)
        self.rateEvol = rateEvolution
        self.massDist = massDistribution
        self.spinDist = spinDistribution
        self.params = self.rateEvol.params+ self.massDist.params+self.spinDist.params
        
    
        self.baseValues = deepcopy(self.rateEvol.baseValues)
        self.baseValues.update(self.massDist.baseValues)
        self.baseValues.update(self.spinDist.baseValues)
        
        self.names = deepcopy(self.rateEvol.names)
        self.names.update(self.massDist.names)
        self.names.update(self.spinDist.names)
        
        # + self.massDist.baseValues.items() + self.spinDist.baseValues.items())
        self.n_params = len(self.params)
    
    
    def log_dR_dm1dm2(self, m1, m2, z, chiEff, lambdaBBH):
        '''log dR/(dm1dm2), correctly normalized  '''
        lambdaBBHrate, lambdaBBHmass, lambdaBBHspin = self._split_lambdas(lambdaBBH)
        theta_rate, theta_mass, theta_spin = self._get_thetas( m1, m2, z, chiEff)
        logdR =  self.rateEvol.log_dNdVdt(theta_rate, lambdaBBHrate)+self.massDist.logpdf(theta_mass, lambdaBBHmass)
        if self.spinDist.__class__.__name__ =='DummySpinDist':
            return logdR
        return logdR+self.spinDist.logpdf(theta_spin, lambdaBBHspin)
        
    
    def _get_thetas(self, m1, m2, z, chiEff):
        '''
        Put here the logic to relate the argument of the distributions to
        m1? m2, z, chi1, chi2
        '''
        theta_rate = z
        theta_mass = m1, m2
        theta_spin= chiEff
        return theta_rate, theta_mass, theta_spin
    
    
    
    def _split_lambdas(self, lambdaBBH):
        '''
        split parameters between R0, Lambda and parameters of the mass function.
        R0, lambda should be the first two parameters in lambdaBBH
        '''
        lambdaBBHrate = lambdaBBH[:self.rateEvol.n_params]
        lambdaBBHmass = lambdaBBH[self.rateEvol.n_params:self.rateEvol.n_params+self.massDist.n_params]
        lambdaBBHspin = lambdaBBH[self.massDist.n_params+self.massDist.n_params:]
        assert len(lambdaBBHspin)==self.spinDist.n_params
        return lambdaBBHrate, lambdaBBHmass, lambdaBBHspin
    
    
    
    def get_base_values(self, params):
        # Should be optimized
        allVals = []
        for obj in (self.rateEvol,  self.massDist, self.spinDist):
            for param in params:
                if param in obj.params:
                    allVals.append(obj.baseValues[param])
        return allVals
    
    def get_labels(self, params):
        # Should be optimized
        allVals = []
        for obj in (self.rateEvol,  self.massDist, self.spinDist):
            for param in params:
                if param in obj.params:
                    allVals.append(obj.names[param])
        return allVals
            
     
    def _set_values(self, values_dict):
            #print('astro basevalues: %s' %str(self.baseValues))
            for obj in (self.rateEvol,  self.massDist, self.spinDist):
                 obj._set_values(values_dict)
            
            # update value also in this object
            for key, value in values_dict.items():
                if key in self.baseValues:
                    self.baseValues[key] = value
                    print('Setting value of %s to %s in %s' %(key, value, self.__class__.__name__))

  