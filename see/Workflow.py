import copy
import inspect
import random

from collections import OrderedDict
import sys
import logging
import numpy as np
import skimage
from skimage import color

from see import ColorSpace, Segmentors

from see.base_classes import param_space, algorithm


class workflow(algorithm):
    
    ##TODO Update to allow paramlist to be either a list or the parameters class
    def __init__(self, algolist=None):
        """Generate algorithm params from parameter list."""
        self.algolist = []        
        self.params = param_space()
        for algo in algolist:
            thisalgo = algo()
            self.algolist.append(thisalgo)
            for key in thisalgo.params:
                self.params[key] = thisalgo.params[key]
            #params.addall(thisalgo.params)
            
    def pipe(self, data):
        for algo in self.algolist:
            data = algo.pipe(data)
        return data
    
#PROBLEMS
# - Next step make a funciton to add parameter spaces.