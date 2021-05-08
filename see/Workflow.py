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

class Workflow(object):

    
    ##TODO Update to allow paramlist to be either a list or the parameters class
    def __init__(self, algolist=None, paramlist=None):
        """Generate algorithm params from parameter list."""
        self.algorithms = algolist
        self.params = parameters()
        if paramlist:
            self.params.fromlist(paramlist)
        else:
            for algo in algolist:
                self.params.add(algo.params)
                self.paramindexes.add(algo.paramindexes)
        self.checkparamindex()

    def checkparamindex(self):
        """Check paramiter index to ensure values are valid"""
        for myparams in self.paramindexes:
            assert myparams in self.params, f"ERROR {myparams} is not in parameter list"
             
    def mutateself(self, flip_prob=0.5):
        """Mutate self and return new params."""
        for myparam in self.paramindexes:
            rand_val = random.random()
            if rand_val < flip_prob:
                self.params[myparam] = random.choice(eval(self.params.ranges[myparam]))
        return self.params
    
    #TODO use name to build a dictionary to use as a chache
    def evaluate(self, img, name=None):
        """Run segmentation algorithm to get inferred mask."""
        return colorspace.getchannel(img, self.params['colorspace'], self.params['channel'])       

    def __str__(self):
        """Return params for algorithm."""
        mystring = f"{self.params['algorithm']} -- \n"
        for p in self.paramindexes:
            mystring += f"\t{p} = {self.params[p]}\n"
        return mystring
    
    def getimag(self):
        """Get the image and use the chache"""
        return self.chache['RGB']  

class parameters(OrderedDict):
    """Construct an ordered dictionary that represents the search space.

    Functions:
    printparam -- returns description for each parameter
    tolist -- converts dictionary of params into list
    fromlist -- converts individual into dictionary of params

    """

    descriptions = OrderedDict()
    ranges = OrderedDict()
    pkeys = []
    
    #0
    #TODO: Change this to the actual strings to make the param space easier to read
    descriptions["colorspace"] = "Pick a colorspace [‘RGB’, ‘HSV’, ‘RGB CIE’, ‘XYZ’, ‘YUV’, ‘YIQ’, ‘YPbPr’, ‘YCbCr’, ‘YDbDr’]"
    ranges["colorspace"] = "['RGB', 'HSV', 'RGB CIE', 'XYZ', 'YUV', 'YIQ', 'YPbPr', 'YCbCr', 'YDbDr']"
    
    #1
    descriptions["channel"] = "A parameter for Picking the Channel 0,1,2"
    ranges["channel"] = "[0,1,2]"
    
    def __init__(self):
        """Set default values for each param in the dictionary."""
        self["multichannel"] = False
        self["colorspace"] = "HSV"
        self["channel"] = 2
        self.pkeys = list(self.keys())

    def printparam(self, key):
        """Return description of parameter from param list."""
        return f"{key}={self[key]}\n\t{self.descriptions[key]}\n\t{self.ranges[key]}\n"

    def __str__(self):
        """Return descriptions of all parameters in param list."""
        out = ""
        for index, k in enumerate(self.pkeys):
            out += f"{index} " + self.printparam(k)
        return out

    def tolist(self):
        """Convert dictionary of params into list of parameters."""
        plist = []
        for key in self.pkeys:
            plist.append(self[key])
        return plist

    def fromlist(self, individual):
        """Convert individual's list into dictionary of params."""
        logging.getLogger().info(f"Parsing Parameter List for {len(individual)} parameters")
        for index, key in enumerate(self.pkeys):
            self[key] = individual[index]


