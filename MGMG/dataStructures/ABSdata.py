#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 14:27:42 2021

@author: Michi
"""
from abc import ABC, abstractmethod



class Data(ABC):
    
    def __init__(self, ):
        pass
        
    @abstractmethod
    def _load_data(self):
        pass
    
    @abstractmethod
    def get_theta(self):
        pass
    
