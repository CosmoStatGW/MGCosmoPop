#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 11:51:38 2021

@author: Michi
"""



class Params(object):

    def __init__(self, dataset_name ):
        
        self.dataset_name=dataset_name
        
        if dataset_name=='mock':
            
            self.allParams = [ 'H0', 'Xi0', 'n',  'lambdaRedshift', 'alpha', 'beta', 'ml', 'sl', 'mh', 'sh' ]
        
            self.trueValues = {'H0':67.74, 
                           'Xi0':1.0, 
                           'n':1.91, 
                           'lambdaRedshift':3.0,
                           'alpha':0.75,
                           'beta':0.0, 
                           'ml':5.0, 
                           'sl':0.1, 
                           'mh':45.0,
                           'sh':0.1}
        
            self.names = {'H0':r'$H_0$', 
                           'Xi0':r'$\Xi_0$', 
                           'n':r'$n$', 
                           'lambdaRedshift':r'$\lambda$',
                           'alpha':r'$\alpha$',
                           'beta':r'$\beta$', 
                           'ml':r'$M_l$', 
                           'sl':r'$\sigma_l$', 
                           'mh':r'$M_h$',
                           'sh':r'$sigma_h$'}
    
        
        else:
            raise NotImplementedError('only mock dataset available.')
            
            
    def get_expected_values(self, params):
        if self.dataset_name=='mock':
            return  [self.trueValues[param] for param in params]
        
    def get_true_values(self, params):
        if self.dataset_name=='mock':
            return  [self.trueValues[param] for param in params]
        else:
            return None
        
    def get_labels(self, params):
        
        return  [self.names[param] for param in params]
        



class PriorLimits(object):
    
    def __init__(self,):
        
        self.limInf = {'H0':20, 
                           'Xi0':0.1, 
                           'n':0, 
                           'lambdaRedshift':-15,
                           'alpha':0,
                           'beta':0.0, 
                           'ml':2, 
                           'sl':0.01, 
                           'mh':20,
                           'sh':0.01}
        self.limSup= {'H0':140, 
                           'Xi0':10, 
                           'n':10, 
                           'lambdaRedshift':10,
                           'alpha':10,
                           'beta':10, 
                           'ml':20, 
                           'sl':1, 
                           'mh':200,
                           'sh':1}
                
        