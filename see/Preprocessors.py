"""This pre-processor designed to do some modification to input data in order to reach a better result.


"""

from collections import OrderedDict
import sys
import logging
import numpy as np
import skimage
from skimage import segmentation
from skimage import color
from skimage import exposure

#TODO in we need to change the param space, to make a new param calss to put it to the pre_params
# such as 
# from see.base_classes import pre_param_space
# TODO, to study the base_classes
from see.base_classes import param_space, algorithm

class pre_params(param_space):
    """Create and add parameters to data structures. (for pre_pro)"""
    
    descriptions = dict()
    ranges = dict()
    pkeys = []
    
    
#This part is the arguments in this class
#define which algorithm we will be using

pre_params.add('algorithm',
               [],
               "string code for the algorithm")
# this part is just for testing, change the parameter base on the array
pre_params.add('nbins',
               [25],#What does this do? what should I change?
               "Number of bins for image histogram"
               )
pre_params.add('mask',
               [None], #those values are based on the range of the array, what should I do?
               "Array of same shape as image, mask==True are used for equalization"
               )

###################
# pre_params.add('nbins',
#                [float(i) / 256 for i in range(0, 256)],#What does this do? what should I change?
#                "Number of bins for image histogram"
#                )

# pre_params.add('mask',
#                [float(i) / 256 for i in range(0, 256)], #those values are based on the range of the array, what should I do?
#                "Array of same shape as image, mask==True are used for equalization"
#                )
###################
#TODO need to add more    
#...


class preprocessor(algorithm):
    """Base class for prepreocessor classes defined below.

    Functions:
    evaluate -- Run preprocessing algorithm to get modified image
    """

    algorithmspace = dict()

    def __init__(self, paramlist=None):
        """Generate algorithm params from parameter list."""


        self.params = pre_params()
    
        #self.params[...] = ...
        #add more param based on the 
        #...suchas
        self.params['nbins'] = 25
        self.params['mask'] = None
        self.params["in_range"] = 'image'
        self.params["out_range"] = 'dtype'
#         self.params['beta1'] = 0.2
#         self.params['beta2'] = 0.7
        self.set_params(paramlist)



    def evaluate(self, img):
        """Run preprocessor algorithm to modify the img."""

        # print(f"Running {self.params}")
        new_img = exposure.equalize_hist(img)
        #TODO need to think about how to actually run the prepro

        sys.stdout.flush()

        self.thisalgo = preprocessor.algorithmspace[self.params['algorithm']](
            self.params)
        return self.thisalgo.evaluate(img)

    
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

# 4/5
class HistEqual(preprocessor):
    """
    Return image after histogram equalization.

    Parameters:

    imagearray:
       - Image array.

    nbinsint: (optional)
       - int, optional
        Number of bins for image histogram. Note: this argument is ignored for integer images, for which each integer is its own bin.

    mask: (optional)
       - ndarray of bools or 0s and 1s
        Array of same shape as image. Only points at which mask == True are used for the equalization, which is applied to the whole image.

    Returns:
        out: float array
    Image array after histogram equalization.

    # example:
    https://scikit-image.org/docs/stable/auto_examples/color_exposure/plot_equalize.html#sphx-glr-auto-examples-color-exposure-plot-equalize-py
    """

    def __init__(self, paramlist=None):
        """Get parameters from parameter list that are used in segmentation algorithm.

        Assign default values to these parameters.
        """
        super(HistEqual, self).__init__(paramlist)
        self.params["algorithm"] = "HistEqual"
        self.params["nbins"] = 0.5
        self.params["mask"] = 0.5

        self.paramindexes = ["nbins", "mask"]
        self.set_params(paramlist)

    def evaluate(self, img):
        """Evaluate pre-pro algorithm on training image.

        Keyword arguments:
        img -- Original training image.

        Output:
        output -- resulting img from algorithm.
        """

        nbins = self.params["nbins"]
        mask = self.params["mask"]
        output = skimage.exposure.equalize_hist(
            image=img,
            nbins=nbins,
            mask=mask
        )
        return output


class Rescale_intensity(preprocessor):
    """
    Return image after stretching or shrinking its intensity levels.

    The desired intensity range of the input and output, in_range and out_range respectively,
    are used to stretch or shrink the intensity range of the input image.

    Parameters:

    imagearray:
       - Image array.

    in_range, out_range:
       - str or 2-tuple, optional

        Min and max intensity values of input and output image.
        The possible values for this parameter are enumerated below.

        ‘image’
         - Use image min/max as the intensity range.

        ‘dtype’
         - Use min/max of the image’s dtype as the intensity range.

        dtype-name
        - Use intensity range based on desired dtype. Must be valid key in DTYPE_RANGE.

        2-tuple
        - Use range_values as explicit min/max intensities.

    Returns:
        out: float array
    Image array after histogram equalization.

    # example:
    https://scikit-image.org/docs/stable/auto_examples/color_exposure/plot_equalize.html#sphx-glr-auto-examples-color-exposure-plot-equalize-py
    """

    def __init__(self, paramlist=None):
        """Get parameters from parameter list that are used in segmentation algorithm.

        Assign default values to these parameters.
        """
        super(Rescale_intensity, self).__init__(paramlist)

        self.params["algorithm"] = "Rescale_intensity"
        self.params["in_range"] = 'image'
        self.params["out_range"] = 'dtype'

        self.paramindexes = ["in_range", "out_range"]
        self.set_params(paramlist)

    def evaluate(self, img):
        """Evaluate pre-pro algorithm on training image.

        Keyword arguments:
        img -- Original training image.

        Output:
        output -- resulting img from algorithm.
        """

        i_range = self.params["in_range"]
        o_range = self.params["out_range"]

        # # Example
        # # Contrast stretching
        # p2, p98 = np.percentile(img, (2, 98))
        # img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))

        output = skimage.exposure.rescale_intensity(
            image=img,
            in_range=i_range,
            out_range=o_range
        )
        return output


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


#segmentor.addsegmentor('QuickShift', QuickShift)

preprocessor.addsegmentor("Histogram Equalization", HistEqual)