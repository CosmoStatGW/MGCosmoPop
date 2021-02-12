#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 16:45:34 2021

@author: Michi
"""
from params import Params



dataset_name='mock'

dataset_name_injections='mock'

fout='runFullTry'


telegramAlert = False

nChains=50
max_n=1000

maxNtaus = 150
checkTauStep = 100


params_inference = [ 'H0', 'lambdaRedshift', 'alpha', 'beta', 'ml', 'sl', 'mh', 'sh']
#'lambdaRedshift', 'alpha', 'beta', 'ml', 'sl', 'mh', 'sh']

marginalise_rate = True
selection_integral_uncertainty = True


###########
# For testing


nSamplesUse= None #100
nObsUse=50
nInjUse=None #100




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


#def get_Lambda(Lambda_test, Lambda_ntest):
    
#    H0, lambdaRedshift, alpha, beta, ml, sl, mh, sh = Lambda_test
#    Xi0, n, R0   = Lambda_ntest
    
#    Lambda = [H0, Xi0, n, R0, lambdaRedshift,  alpha, beta, ml, sl, mh, sh]
#    return Lambda