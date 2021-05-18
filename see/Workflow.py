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

    worklist = []

    @classmethod
    def addalgos(cls, algo):
        if type(algo) == list:
            for a in algo:
                workflow.worklist.append(a)
        else:  
            workflow.worklist.append(algo)
        
    def __init__(self, paramlist=None):
        """Generate algorithm params from parameter list."""
        self.params = param_space()
        self.set_params(paramlist)
        for algo in workflow.worklist:
            thisalgo = algo()
            self.params.addall(thisalgo.params)
        self.set_params(paramlist)

            
    def mutateself(self, flip_prob=0.5):
        """Mutate self and return new params."""
        print("using workflow mutate algorithm and looping over workflow")
        for algo in workflow.worklist:
            thisalgo = algo()
            thisalgo.params.addall(thisalgo.params)
            
            thisalgo.mutateself(flip_prob=flip_prob)
            self.params.addall(thisalgo.params)

    def pipe(self, data):
        for algo_constructor in workflow.worklist:
            algo = algo_constructor(self.params)
            algo.params.addall(self.params)
            data = algo.pipe(data)
        return data
