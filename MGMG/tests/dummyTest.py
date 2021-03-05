#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 09:31:38 2021

@author: Michi
"""

import os
import sys

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))



def main():
    
    from population.astro.astroMassDistribution import AstroSmoothPowerLawMass

    import Globals
    
    powLaw = AstroSmoothPowerLawMass()
    print(powLaw.params)
    
    print(Globals.dataPath)
    
    
    
if __name__=='__main__':
    
    main()