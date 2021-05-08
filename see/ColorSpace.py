import copy
import inspect
import random

from collections import OrderedDict
import sys
import logging
import numpy as np
import skimage
from skimage import color
from see.base_classes import algorithm

def __init__(slef):
    print("making a Colorspace")

class colorspace(algorithm):
    
    def getchannel(img, colorspace, channel):
        """function that returns a single channel from an image
        ['RGB', ‘HSV’, ‘RGB CIE’, ‘XYZ’, ‘YUV’, ‘YIQ’, ‘YPbPr’, ‘YCbCr’, ‘YDbDr’]
        """
        dimention=3;
        if (len(img.shape) == 2):
            c_img = img.copy();
            img = np.zeros([c_img.shape[0], c_img.shape[1],3])
            img[:,:,0] = c_img;
            img[:,:,1] = c_img;
            img[:,:,2] = c_img;
            return [img, c_img, 1]

        if(colorspace == 'RGB'):
            return [img, img[:,:,channel], 3]
        else:
            space = color.convert_colorspace(img, 'RGB', colorspace)
            return [space, space[:,:,channel], 3]
    
    ##TODO Update to allow paramlist to be either a list or the parameters class
    def __init__(self, paramlist=None):
        """Generate algorithm params from parameter list."""
        self.params = parameters()
        
        self.params.add('colorspace', 
                        "['RGB', 'HSV', 'RGB CIE', 'XYZ', 'YUV', 'YIQ', 'YPbPr', 'YCbCr', 'YDbDr']",
                        "Pick a colorspace [‘RGB’, ‘HSV’, ‘RGB CIE’, ‘XYZ’, ‘YUV’, ‘YIQ’, ‘YPbPr’, ‘YCbCr’, ‘YDbDr’]"
                       )
        self.params['colorspace'] = 'RGB'
        
        self.params.add('channel",
                        "[0,1,2]"
                        "A parameter for Picking the Channel 0,1,2"
                       )
        self.params['channel'] = 2
        
        self.chache = dict()
        if paramlist:
            self.params.fromlist(paramlist)
        else:
            self.params["multichannel"] = True
            self.params["colorspace"] = "RGB"
            self.params["channel"] = 2
        self.paramindexes = ["multichannel", "colorspace", "channel"]
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
