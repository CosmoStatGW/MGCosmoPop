#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 16:45:34 2021

@author: Michi
"""
from params import Params



dataset_name='mock'

dataset_name_injections='mock'

fout='runFullLCDM'

telegramAlert = False


params_inference = [ 'H0', 'Om0', 'Xi0',  'lambdaRedshift', 'alpha', 'beta', 'ml', 'sl', 'mh', 'sh']

nChains=50 #2*len(params_inference)
max_n=10000

maxNtaus = 150
checkTauStep = 100

perc_variation_init = 30

marginalise_rate = True
selection_integral_uncertainty = True

nPools=5


###########
# For testing

nSamplesUse= None
nObsUse=None
nInjUse=None




#####################
# Don't touch the part below
#####################


myParams = Params(dataset_name)
params_n_inference = [param for param in myParams.allParams if param not in params_inference]


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