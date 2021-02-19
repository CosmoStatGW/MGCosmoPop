#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 16:45:34 2021

@author: Michi
"""
from params import Params, PriorLimits



dataset_name='mock'

dataset_name_injections='mock' # LVC
ifar_th = 1.

fout='runFullR0testPickle'

telegramAlert = False


params_inference = [ 'H0', 'Om0', 'Xi0', 'R0','lambdaRedshift', 'alpha', 'beta', 'ml', 'sl', 'mh', 'sh']


priors_types = {'Om0':'gauss' , 'R0': 'flatLog'} 
priors_params = { 'Om0': {'mu': 0.3 , 'sigma':0.001} }

nChains=25 #2*len(params_inference)
max_n=300

maxNtaus = 100
checkTauStep = 100

perc_variation_init = 30

marginalise_rate = False
selection_integral_uncertainty = True

nPools=4

mass_model='Farr' # Farr
mass_normalization = 'integral'  # integral #pivot
# Value of the normalized pdf p(m1, m2) at the pivot scale
pivot_val=64.4

###########
# For testing

nSamplesUse= 100
nObsUse=10
nInjUse=100

verbose_bias = False



#####################
# Don't touch the part below
#####################

with_jacobian_inj=True
if dataset_name_injections=='LVC':
    with_jacobian_inj = False



myParams = Params(dataset_name)
params_n_inference = [param for param in myParams.allParams if param not in params_inference]
if mass_normalization=='pivot' and 'R0' in params_inference:
    print('Normalizing mass function at the pivolt scale of 30')
    myParams.reset_fiducial('R0', pivot_val)
    myParams.reset_name('R0', r'$R_{30}$')


allMyPriors = PriorLimits()
allMyPriors.set_priors(priors_types=priors_types, priors_params=priors_params)
myPriorLimits = allMyPriors.priorLimits(params_inference)

lines=[]
lines.append('def get_Lambda(Lambda_test, Lambda_ntest):')
lines.append('    '+ (', ').join(params_inference) + ' = Lambda_test')
lines.append('    '+ (', ').join(params_n_inference) + ' = Lambda_ntest')
lines.append('    Lambda = ['+ (', ').join(myParams.allParams) +']')
lines.append('    return Lambda' )

with open('getLambda.py', 'w') as f:
    f.write(('\n').join(lines))


#####
# Template for function getLambda

#def get_Lambda(Lambda_test, Lambda_ntest):
    
#    H0, lambdaRedshift, alpha, beta, ml, sl, mh, sh = Lambda_test
#    Xi0, n, R0   = Lambda_ntest
    
#    Lambda = [H0, Xi0, n, R0, lambdaRedshift,  alpha, beta, ml, sl, mh, sh]
#    return Lambda