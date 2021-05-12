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

    algolist = []

    def addalgo(self, algorithm):
        workflow.algolist.append(algorithm)

        thisalgo = algorithm()
        self.params.addall(thisalgo.params)

    def __init__(self, paramlist=None, algolist=None):
        """Generate algorithm params from parameter list."""
        self.params = param_space()
        if paramlist:
            self.params.fromlist(paramlist)
        if algolist:
            for algo in algolist:
                self.addalgo(algo)

    def pipe(self, data):
        for constructor in self.algolist:
            algo = constructor(paramlist=self.params)
            data = algo.pipe(data)
        return data

# PROBLEMS
# - Next step make a funciton to add parameter spaces.
