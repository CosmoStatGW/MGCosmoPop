#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 17:37:48 2021

@author: Michi
"""
import astropy.units as u
from astropy.cosmology import Planck15
from astropy import constants as const
import os


which_unit=u.Mpc

H0GLOB = Planck15.H0.value # In km/s/Mpc
Om0GLOB = Planck15.Om0
Xi0GLOB = 1.
nGLOB = 1.91


clight=const.c.value*1e-03 # c in km/s



dirName  = os.path.join(os.path.dirname(os.path.abspath(__file__)))
dataPath=os.path.join(dirName, 'data')

