"""This pre-processor designed to do some modification to input data in order to reach a better result.


"""

from collections import OrderedDict
import sys
import logging
import numpy as np
import skimage
from skimage import segmentation
from skimage import color


#TODO in we need to change the param space, to make a new param calss to put it to the pre_params
# such as 
# from see.base_classes import pre_param_space
# TODO, to study the base_classes
from see.base_classes import param_space, algorithm

class pre_params(pre_param_space):
    """Create and add parameters to data structures. (for pre_pro)"""
    
    descriptions = dict()
    ranges = dict()
    pkeys = []
    
    
#This part is the arguments in this class
#define which algorithm we will be using
#The param is not for preprocessor, just showing what I should do
pre_params.add('algorithm',
               [],
               "string code for the algorithm")

pre_params.add('alpha1',
               [float(i) / 256 for i in range(0, 256)],#What does this do?
               "General Purpos Lower bound threshold"
               )

pre_params.add('alpha2',
               [float(i) / 256 for i in range(0, 256)],
               "General Purpos Upper bound threshold"
               )
#TODO need to add more    
#...


class preprocessor(pre_algorithm):
    """Base class for prepreocessor classes defined below.

    Functions:
    #TODO Add the description for the methdos 

    """

    algorithmspace = dict()

    def __init__(self, paramlist=None):
        """Generate algorithm params from parameter list."""

        self.params = pre_params()
    
        self.params[...] = ...
        #add more param based on the 
        #...suchas
#         self.params['alpha1'] = 0.3
#         self.params['alpha2'] = 0.5
#         self.params['beta1'] = 0.2
#         self.params['beta2'] = 0.7
        self.set_params(paramlist)



    def evaluate(self, img):
        """Run preprocessor algorithm to modify the img."""
        #TODO need to think about how to actually run the prepro
        /*
        sys.stdout.flush()
        #is this passing in the params based on what
        self.thisalgo = segmentor.algorithmspace[self.params['algorithm']](
            self.params)
        return self.thisalgo.evaluate(img)
        */
    
    #TODO what does this do?
    def pipe(self, data):
        """Set data.mask to an evaluated image."""
        data.mask = self.evaluate(data.img)
        return data
    
    #what does this do?
    @classmethod
    #TODO cls stands for "class"
    def addsegmentor(cls, key, pre):
        """Add a preprocessor."""
        pre_params.ranges['algorithm'].append(key)
        #TODO change cls
        cls.algorithmspace[key] = pre


#..
#TODO create different classes for each algo 
#such as 



class QuickShift(segmentor):
    """Perform the Quick Shift segmentation algorithm.
    
    Segments images with quickshift
    clustering in Color (x,y) space. Returns ndarray segmentation mask of the labels.
    Parameters:
    image -- ndarray, input image
    ratio -- float, balances color-space proximity & image-space
        proximity. Higher vals give more weight to color-space
    kernel_size: float, Width of Guassian kernel using smoothing.
        Higher means fewer clusters
    max_dist -- float, Cut-off point for data distances. Higher means fewer clusters
    sigma -- float, Width of Guassian smoothing as preprocessing.
        Zero means no smoothing
    random_seed -- int, Random seed used for breacking ties.
    https://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.quickshift
    """

    def __init__(self, paramlist=None):
        """Get parameters from parameter list that are used in segmentation algorithm.
        
        Assign default values to these parameters.
        """
        super(QuickShift, self).__init__(paramlist)
        self.params["algorithm"] = "QuickShift"
        self.params["alpha1"] = 0.5
        self.params["beta1"] = 0.5
        self.params["beta2"] = 0.5
        self.paramindexes = ["alpha1", "beta1", "beta2"]
        self.set_params(paramlist)

    def evaluate(self, img):
        """Evaluate segmentation algorithm on training image.

        Keyword arguments:
        img -- Original training image.

        Output:
        output -- resulting segmentation mask from algorithm.
        """
        mindim = min(img.shape)
        mindim = min([mindim,100])
        ratio = self.params["alpha1"]
        kernel_size = mindim / 10 * self.params["beta1"] + 1
        
        max_dist = mindim * self.params["beta2"] + 1
        output = skimage.segmentation.quickshift(
            img,
            ratio=ratio,
            kernel_size=kernel_size,
            max_dist=max_dist,
            sigma=0,  # TODO this should be handeled in the preprocessing step
            #random_seed=1,
            convert2lab=False
        )
        return output


segmentor.addsegmentor('QuickShift', QuickShift)