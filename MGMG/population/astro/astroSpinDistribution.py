#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 09:58:40 2021

@author: Michi
"""

from ..ABSpopulation import BBHDistFunction
import numpy as np

########################################################################
# SPIN DISTRIBUTION
########################################################################

class DummySpinDist(BBHDistFunction):
    
    def __init__(self, ):
        BBHDistFunction.__init__(self)
    
    def logpdf(theta, lambdaBBHmass):
        chi1, chi2 =theta
        return np.zeros(chi1.shape[0])