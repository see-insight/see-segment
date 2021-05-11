"""Segmentor algorithm library designed to segment images with a searchable parameter space.
 This libary actually does not incode the search code itself, instead it just defines
  the search parameters and the evaluation funtions."""

import copy
import inspect
import random

from collections import OrderedDict
import sys
import logging
import numpy as np
import skimage
from skimage import segmentation
from skimage import color

from see.Segment_Similarity_Measure import FF_ML2DHD_V2 as FitnessFunction

from see.base_classes import param_space, algorithm


class seg_params(param_space):
    descriptions = dict()
    ranges = dict()
    pkeys = []


seg_params.add('algorithm',
               [],
               "string code for the algorithm")

seg_params.add('alpha1',
               [float(i)/256 for i in range(0, 256)],
               "General Purpos Lower bound threshold"
               )

seg_params.add('alpha2',
               [float(i)/256 for i in range(0, 256)],
               "General Purpos Upper bound threshold"
               )

seg_params.add('beta1',
               [float(i)/256 for i in range(0, 256)],
               "General Purpos Lower bound threshold"
               )

seg_params.add('beta2',
               [float(i)/256 for i in range(0, 256)],
               "General Purpos Upper bound threshold"
               )

seg_params.add('gamma1',
               [float(i)/256 for i in range(0, 256)],
               "General Purpos Lower bound threshold"
               )

seg_params.add('gamma2',
               [float(i)/256 for i in range(0, 256)],
               "General Purpos Upper bound threshold"
               )

seg_params.add('n_segments',
               [i for i in range(0, 10)],
               "General Purpos Upper bound threshold"
               )

seg_params.add('max_iter',
               [i for i in range(1, 20)],
               "General Purpos Upper bound threshold"
               )


class segmentor(algorithm):
    """Base class for segmentor classes defined below.

    Functions:
    evaluate -- Run segmentation algorithm to get inferred mask.

    """

    algorithmspace = dict()

    def __init__(self, paramlist=None):
        """Generate algorithm params from parameter list."""
        #super(ColorThreshold, self).__init__(paramlist)
        self.params = seg_params()

        self.params['algorithm'] = 'ColorThreshold'
        self.params['alpha1'] = 0.3
        self.params['alpha2'] = 0.5
        self.params['beta1'] = 0.2
        self.params['beta2'] = 0.7
        self.params['gamma1'] = 0.3
        self.params['gamma2'] = 0.5
        self.params['n_segments'] = 2
        self.params['max_iter'] = 10

    # TODO use name to build a dictionary to use as a chache
    def evaluate(self, img):
        """Run segmentation algorithm to get inferred mask."""
        self.thisalgo = segmentor.algorithmspace[self.params['algorithm']](
            self.params)
        return self.thisalgo.evaluate(img)

    def pipe(self, data):
        data.mask = self.evaluate(data.img)
        return data

    @classmethod
    def addsegmentor(cls, key, seg):
        seg_params.ranges['algorithm'].append(key)
        cls.algorithmspace[key] = seg


class ColorThreshold(segmentor):
    """ColorThreshold

    Peform Color Thresholding segmentation algorithm. Segments parts of the image
    based on the numerical values for the respective channel.

    Parameters:
    mulitchannel - (multichannel) - bool, Whether the image is 2D or 3D
    colorspace - (colorspace) Select the colorspace [‘RGB’, ‘HSV’, ‘RGB CIE’, ‘XYZ’, ‘YUV’, ‘YIQ’, ‘YPbPr’, ‘YCbCr’, ‘YDbDr’]
    channel - (channel) color chanel (0:R/H/L 1:G/S/A, 2:B/V/B)
    ch0_mn - (alpha1) - minimum thresholding value for channel 0
    ch0_mx - (alpha2) - maximum thresholding value for channel 0
    ch1_mn - (beta1) - minimum thresholding value for channel 1
    ch1_mx - (beta2) - maximum thresholding value for channel 1
    ch2_mn - (gamma1) - minimum thresholding value for channel 2
    ch2_mx - (gamma2) - maximum thresholding value for channel 2

    Note: a colorspace of 'HSV' and a channel of 2 is a grayscale image. 

    Typically any pixel between my_mn and my_mx are true. Other pixels are false.

    if my_mn > my_mx then the logic flips and anything above my_mn and below my_mx are true. 
    The pixels between the valuse are false
    """

    def __init__(self, params=None):
        """Get parameters from parameter list that are used in segmentation algorithm.
         Assign default values to these parameters."""
        self.params = seg_params()

        self.params["algorithm"] = "ColorThreshold"
        self.params["alpha1"] = 0.4
        self.params["alpha2"] = 0.6
        self.params["beta1"] = 0.4
        self.params["beta2"] = 0.6
        self.params["gamma1"] = 0.4
        self.params["gamma2"] = 0.6
        self.paramindexes = ["alpha1", "alpha2",
                             "beta1", "beta2",
                             "gamma1", "gamma2"]
        if params:
            if (type(params) == list):
                self.params.fromlist(params)
            else:
                self.params = params
        # TODO I think we want this function butit is causing a bug
        self.checkparamindex()

    def evaluate(self, img):  # XX
        """Evaluate segmentation algorithm on training image.

        Keyword arguments:
        img -- Original training image.

        Output:
        output -- resulting segmentation mask from algorithm.

        """
        minlist = ["alpha1", "beta1", "gamma1"]
        maxlist = ["alpha2", "beta2", "gamma2"]

        output = None

        if (len(img.shape) > 2):
            output = np.ones([img.shape[0], img.shape[1]])
            for dimidx in range(3):
                pscale = np.max(img[:, :, dimidx])
                my_mn = self.params[minlist[dimidx]] * pscale
                my_mx = self.params[maxlist[dimidx]] * pscale

                if my_mn < my_mx:
                    output[img[:, :, dimidx] < my_mn] = 0
                    output[img[:, :, dimidx] > my_mx] = 0
                else:
                    flag1 = img[:, :, dimidx] > my_mn
                    flag2 = img[:, :, dimidx] < my_mx
                    output[np.logical_and(flag1, flag2)] = 0
        else:
            pscale = np.max(img)
            if "channel" in self.params:
                chidx = self.params["channel"]
            else:
                chidx = 0
            my_mx = self.params[maxlist[chidx]] * pscale
            my_mn = self.params[minlist[chidx]] * pscale

            if my_mn < my_mx:
                output = np.ones(img.shape)
                output[img < my_mn] = 0
                output[img > my_mx] = 0
            else:
                output = np.zeros(img.shape)
                output[img > my_mn] = 1
                output[img < my_mx] = 1
        return output


segmentor.addsegmentor('ColorThreshold', ColorThreshold)


class Felzenszwalb(segmentor):
    """Perform Felzenszwalb segmentation algorithm. The felzenszwalb algorithms computes a 
    graph based on the segmentation. Produces an oversegmentation of the multichannel using 
    min-span tree. Returns an integer mask indicating the segment labels.

    Note: a colorspace of 'HSV' and a channel of 2 is a grayscale image. 

    https://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.felzenszwalb

    Parameters:
    colorspace - (colorspace) Select the colorspace (0:RGB, 1:HSV, 2:LAB)
    channel - (channel) color chanel (0:R/H/L 1:G/S/A, 2:B/V/B)
    scale - (alpha2*1000) - float, higher meanse larger clusters
    sigma - (alpha1) - float, std. dev of Gaussian kernel for preprocessing
    min_size - int(beta1*100) - int, minimum component size (in pixels). For postprocessing
    """

#     def __doc__(self):
#         """Return help string for function."""
#         myhelp = "Wrapper function for the scikit-image Felzenszwalb segmentor:"
#         myhelp += f" xx {skimage.segmentation.random_walker.__doc__}"
#         return myhelp

    def __init__(self, paramlist=None):
        """Get parameters from parameter list that are used in segmentation algorithm.
         Assign default values to these parameters."""
        super(Felzenszwalb, self).__init__(paramlist)
        self.params["algorithm"] = "Felzenszwalb"
        self.params["alpha2"] = 0.984
        self.params["alpha1"] = 0.09
        self.params["beta1"] = 0.92
        self.paramindexes = ["alpha1", "alpha2", "beta1"]
        self.checkparamindex()

    def evaluate(self, img):
        """Evaluate segmentation algorithm on training image.

        Keyword arguments:
        img -- Original training image.

        Output:
        output -- resulting segmentation mask from algorithm.

        """

        scale = self.params["alpha2"]*1000
        sigma = self.params["alpha1"]
        min_size = int(self.params["beta1"]*100)

        if (len(img.shape) > 2):
            output = skimage.segmentation.felzenszwalb(
                img,
                scale,
                sigma,
                min_size,
                multichannel=True
            )
        else:
            output = skimage.segmentation.felzenszwalb(
                img,
                scale,
                sigma,
                min_size,
                multichannel=False
            )

        return output

#     def sharepython(self, img):


#         multichannel = self.params['multichannel']
#         if len(img.shape) == 1:
#             multichannel = False
#         if (multichannel):
#             mystring= f"""
#             output = skimage.segmentation.felzenszwalb(
#                 img,
#                 {self.params["alpha2"]*1000},
#                 {self.params["alpha1"]},
#                 {int(self.params["beta1"]*100)},
#                 multichannel={multichannel}
#             )"""
#         else:
#             mystring= f"""
#             output = skimage.segmentation.felzenszwalb(
#                 getchannel(img, {self.params["channel"]}),
#                 {self.params["alpha2"]*1000},
#                 {self.params["alpha1"]},
#                 {int(self.params["beta1"]*100)},
#                 multichannel={multichannel}
#             )"""
#         return mystring

segmentor.addsegmentor('Felzenszwalb', Felzenszwalb)


class Slic(segmentor):
    """Perform the Slic segmentation algorithm. Segments k-means clustering in Color space
     (x, y, z). Returns a 2D or 3D array of labels.

    Parameters:
    image -- ndarray, input image
    n_segments -- int, approximate number of labels in segmented output image
    compactness -- float, Balances color proximity and space proximity.
        Higher values mean more weight to space proximity (superpixels
        become more square/cubic) Recommended log scale values (0.01,
        0.1, 1, 10, 100, etc)
    max_iter -- int, max number of iterations of k-means
    sigma -- float or (3,) shape array of floats, width of Guassian
        smoothing kernel. For pre-processing for each dimesion of the
        image. Zero means no smoothing.
    spacing -- (3,) shape float array. Voxel spacing along each image
        dimension. Defalt is uniform spacing
    multichannel -- bool,  multichannel (True) vs grayscale (False)

    enforce_connectivity

   https://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.slic

   https://www.pyimagesearch.com/2014/07/28/a-slic-superpixel-tutorial-using-python/


    """

    def __init__(self, paramlist=None):
        """Get parameters from parameter list that are used in segmentation algorithm.
         Assign default values to these parameters."""
        super(Slic, self).__init__(paramlist)
        self.params["algorithm"] = "Slic"
        self.params["n_segments"] = 5
        self.params["beta1"] = 2
        self.params["max_iter"] = 10
        self.params["alpha1"] = 0.5
        self.paramindexes = ["n_segments", "alpha1", "beta1", "max_iter"]
        self.checkparamindex()
        self.slico = False

    def evaluate(self, img):
        """Evaluate segmentation algorithm on training image.

        Keyword arguments:
        img -- Original training image.

        Output:
        output -- resulting segmentation mask from algorithm.

        """

        compactness = 10**(self.params["beta1"]*3-3)
        n_segments = self.params["n_segments"]+1
        max_iter = self.params["max_iter"]
        if (len(img.shape) > 2):
            output = skimage.segmentation.slic(
                img,
                n_segments=n_segments,
                compactness=compactness,
                max_iter=max_iter,
                # Gaussian smoothing should happen as a preprocessing step.
                sigma=0,
                convert2lab=False,
                multichannel=True,
                slic_zero=self.slico
            )
        else:
            output = skimage.segmentation.slic(
                img,
                n_segments=n_segments,
                compactness=compactness,
                max_iter=max_iter,
                sigma=self.params["alpha1"],
                multichannel=False,
                slic_zero=self.slico
            )
        return output


segmentor.addsegmentor('Slic', Slic)

# TODO Update to remove any parameters that SLICO dosn't use. (Currently this includes the SLIP parameters)


class SlicO(Slic):
    """Perform the SlicO segmentation algorithm. Segments k-means clustering in Color space
     (x, y, z). Returns a 2D or 3D array of labels.

    Parameters:
    image -- ndarray, input image
    n_segments -- int, approximate number of labels in segmented output image
    compactness -- float, Balances color proximity and space proximity.
        Higher values mean more weight to space proximity (superpixels
        become more square/cubic) Recommended log scale values (0.01,
        0.1, 1, 10, 100, etc)
    max_iter -- int, max number of iterations of k-means
    sigma -- float or (3,) shape array of floats, width of Guassian
        smoothing kernel. For pre-processing for each dimesion of the
        image. Zero means no smoothing.
    spacing -- (3,) shape float array. Voxel spacing along each image
        dimension. Defalt is uniform spacing
    multichannel -- bool,  multichannel (True) vs grayscale (False)

    enforce_connectivity

    https://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.slic

    https://www.pyimagesearch.com/2014/07/28/a-slic-superpixel-tutorial-using-python/


    """

    def __init__(self, paramlist=None):
        """Get parameters from parameter list that are used in segmentation algorithm.
         Assign default values to these parameters."""
        super(SlicO, self).__init__(paramlist)
        self.slico = True


segmentor.addsegmentor('SlicO', SlicO)

# TODO Quickshift is very slow, we need to do some benchmarks and see what are resonable running ranges.


class QuickShift(segmentor):
    """Perform the Quick Shift segmentation algorithm. Segments images with quickshift
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
         Assign default values to these parameters."""
        super(QuickShift, self).__init__(paramlist)
        self.params["algorithm"] = "QuickShift"
        self.params["colorspace"] = "HSV"
        self.params["alpha1"] = 0.5
        self.params["beta1"] = 0.5
        self.params["beta2"] = 0.5

        self.paramindexes = ["alpha1", "beta1", "beta2"]
        self.checkparamindex()

    def evaluate(self, img):
        """Evaluate segmentation algorithm on training image.

        Keyword arguments:
        img -- Original training image.

        Output:
        output -- resulting segmentation mask from algorithm.

        """

        mindim = min(img.shape)

        ratio = self.params["alpha1"]
        kernel_size = mindim/10*self.params["beta1"]+1
        max_dist = mindim*self.params["beta2"]
        output = skimage.segmentation.quickshift(
            img,
            ratio=ratio,
            kernel_size=kernel_size,
            max_dist=max_dist,
            sigma=0,  # TODO this should be handeled in the preprocessing step
            random_seed=1,
        )
        return output


segmentor.addsegmentor('QuickShift', QuickShift)

# TODO Watershed one seems to be broken all we get is a line at the top.


class Watershed(segmentor):
    """Perform the Watershed segmentation algorithm. Uses user-markers.
     treats markers as basins and 'floods' them. Especially good if overlapping objects.
      Returns a labeled image ndarray.

    Parameters:
    image -- ndarray, input array
    compactness -- float, compactness of the basins. Higher values
        make more regularly-shaped basin.

    https://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.watershed

    """

    # Not using connectivity, markers, or offset params as arrays would
    # expand the search space too much.
    # abbreviation for algorithm = WS

    def __init__(self, paramlist=None):
        """Get parameters from parameter list that are used in segmentation algorithm.
         Assign default values to these parameters."""
        super(Watershed, self).__init__(paramlist)
        self.params["algorithm"] = "Watershed"
        self.params["alpha1"] = 0.66
        self.paramindexes = ["alpha1"]
        self.checkparamindex()

    def evaluate(self, img):
        """Evaluate segmentation algorithm on training image.

        Keyword arguments:
        img -- Original training image.

        Output:
        output -- resulting segmentation mask from algorithm.



        """

        compactness = self.params["alpha1"]*3

        output = skimage.segmentation.watershed(
            img, markers=None, compactness=compactness
        )
        return output


segmentor.addsegmentor('Watershed', Watershed)

# TODO Chan_Vese one seems very broken.  All we get is a circle.


class Chan_Vese(segmentor):
    """Peform Chan Vese segmentation algorithm. ONLY GRAYSCALE. Segments objects
     without clear boundaries. Returns segmentation array of algorithm.

    Parameters:
    image -- ndarray grayscale image to be segmented
    mu -- float, 'edge length' weight parameter. Higher mu vals make a
        'round edge' closer to zero will detect smaller objects. Typical
        values are from 0 - 1.
    lambda1 -- float 'diff from average' weight param to determine if
        output region is True. If lower than lambda1, the region has a
        larger range of values than the other
    lambda2 -- float 'diff from average' weight param to determine if
        output region is False. If lower than lambda1, the region will
        have a larger range of values
    tol -- positive float, typically (0-1), very low level set variation
        tolerance between iterations.
    max_iter -- uint,  max number of iterations before algorithms stops
    dt -- float, Multiplication factor applied at the calculations step



    """

    # Abbreviation for Algorithm = CV

    def __init__(self, paramlist=None):
        """Get parameters from parameter list that are used in segmentation algorithm.
         Assign default values to these parameters."""
        super(Chan_Vese, self).__init__(paramlist)
        self.params["algorithm"] = "Chan_Vese"
        self.params["alpha1"] = 1
        self.params["beta1"] = 1
        self.params["beta2"] = 1
        self.params["max_iter"] = 10
        self.params["alpha2"] = 0.10
        self.params["n_segments"] = 0
        # self.params["tolerance"] = 0.001 #TODO Removed, consider adding in later if need be.
        self.paramindexes = ["alpha1", "alpha2",
                             "beta1", "beta2", "n_segments", "max_iter"]
        self.checkparamindex()

    def evaluate(self, img):
        """Evaluate segmentation algorithm on training image.

        Keyword arguments:
        img -- Original training image.

        Output:
        output -- resulting segmentation mask from algorithm.

        """

        # TODO I think this should be between zero and one.
        mu = self.params["alpha1"]*2
        # TODO Not sure about the range of these. Previous was (10,20)
        lambda1 = self.params["beta1"]
        lambda2 = self.params["beta2"]
        max_iter = self.params["max_iter"]
        dt = self.params["alpha2"]

        level_set_shapes = ['checkerboard', 'disk', 'small disk']
        init_level_set = level_set_shapes[self.params['n_segments'] % 3]

        if(len(img.shape) > 2):
            if "channel" in self.params:
                channel = self.params['channel']
                img = img[:, :, channel]
            else:
                img = color.rgb2gray(img)

        output = skimage.segmentation.chan_vese(
            img,
            mu=mu,
            lambda1=lambda1,
            lambda2=lambda2,
            max_iter=max_iter,
            dt=dt,
            init_level_set=init_level_set

        )
        return output


segmentor.addsegmentor('Chan_Vese', Chan_Vese)


class Morphological_Chan_Vese(segmentor):
    """Peform Morphological Chan Vese segmentation algorithm.
     ONLY WORKS ON GRAYSCALE. Active contours without edges. Can be used to
      segment images/volumes without good borders. Required that the inside of
       the object looks different than outside (color, shade, darker).

    Parameters:
    image -- ndarray of grayscale image
    iterations -- uint, number of iterations to run
    init_level_set -- str, or array same shape as image. Accepted string
        values are:
        'checkerboard': Uses checkerboard_level_set. Returns a binary level set of a checkerboard
        'circle': Uses circle_level_set. Creates a binary level set of a circle, given radius and a
            center
    smoothing -- uint, number of times the smoothing operator is applied
        per iteration. Usually around 1-4. Larger values make it smoother
    lambda1 -- Weight param for outer region. If larger than lambda2,
        outer region will give larger range of values than inner value.
    lambda2 -- Weight param for inner region. If larger thant lambda1,
        inner region will have a larger range of values than outer region.

    https://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.morphological_chan_vese

    """

    # Abbreviation for algorithm = MCV

    def __init__(self, paramlist=None):
        """Get parameters from parameter list that are used in segmentation algorithm.
         Assign default values to these parameters."""
        super(Morphological_Chan_Vese, self).__init__(paramlist)
        if not paramlist:
            self.params["algorithm"] = "Morphological_Chan_Vese"
            self.params["alpha1"] = 1
            self.params["beta1"] = 1
            self.params["beta2"] = 1
            self.params["max_iter"] = 10
            self.params["n_segments"] = 0
            # self.params["tolerance"] = 0.001 #TODO Removed, consider adding in later if need be.
        self.paramindexes = ["alpha1",  "beta1",
                             "beta2", "n_segments", "max_iter"]
        self.checkparamindex()

    def evaluate(self, img):
        """Evaluate segmentation algorithm on training image.

        Keyword arguments:
        img -- Original training image.

        Output:
        output -- resulting segmentation mask from algorithm.

        """

        # TODO We may want to move this? We need a number 1-4 smoothing iterations
        smoothing = int(self.params["alpha1"]*4)

        # TODO Not sure about the range of these. Previous was (10,20)
        lambda1 = self.params["beta1"]
        lambda2 = self.params["beta2"]
        max_iter = self.params["max_iter"]
        level_set_shapes = ['checkerboard', 'circle']
        init_level_set = level_set_shapes[self.params['n_segments'] % 2]

        if(len(img.shape) > 2):
            if "channel" in self.params:
                channel = self.params['channel']
                img = img[:, :, channel]
            else:
                img = color.rgb2gray(img)

        output = skimage.segmentation.morphological_chan_vese(
            img,
            iterations=max_iter,
            init_level_set=init_level_set,
            smoothing=smoothing,
            lambda1=lambda1,
            lambda2=lambda2,
        )
        return output


segmentor.addsegmentor('Morphological_Chan_Vese', Morphological_Chan_Vese)


class MorphGeodesicActiveContour(segmentor):
    """Peform Morphological Geodesic Active Contour segmentation algorithm. Uses
     an image from inverse_gaussian_gradient in order to segment object with visible,
      but noisy/broken borders. inverse_gaussian_gradient computes the magnitude of
       the gradients in an image. Returns a preprocessed image suitable for above function.
        Returns ndarray of segmented image.

    Parameters:
    gimage -- array, preprocessed image to be segmented.
    iterations -- uint, number of iterations to run.
    init_level_set -- str, array same shape as gimage. If string, possible
        values are:
        'checkerboard': Uses checkerboard_level_set. Returns a binary level set of a checkerboard
        'circle': Uses circle_level_set. Creates a binary level set of a circle, given radius and a
            center
    smoothing -- uint, number of times the smoothing operator is applied
        per iteration. Usually 1-4, larger values have smoother segmentation.
    threshold -- Areas of image with a smaller value than the threshold are borders.
    balloon -- float, guides contour of low-information parts of image.

    morphological_geodesic_active_contour
    https://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.morphological_geodesic_active_contour

    Preprocessign step:
    https://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.inverse_gaussian_gradient


    """

    # Abbrevieation for algorithm = AC

    def __init__(self, paramlist=None):
        """Get parameters from parameter list that are used in segmentation algorithm.
         Assign default values to these parameters."""
        super(MorphGeodesicActiveContour, self).__init__(paramlist)
        if not paramlist:
            self.params["algorithm"] = "MorphGeodesicActiveContour"
            self.params["alpha1"] = 1
            self.params["alpha2"] = 1
            self.params["beta1"] = 0.2
            self.params["beta2"] = 0.3
            self.params["beta2"] = 1
            self.params["max_iter"] = 10
            self.params["n_segments"] = 0
            # self.params["tolerance"] = 0.001 #TODO Removed, consider adding in later if need be.
        self.paramindexes = ["alpha1",  "alpha2",
                             "beta1", "beta2", "n_segments", "max_iter"]
        self.checkparamindex()

    def evaluate(self, img):
        """Evaluate segmentation algorithm on training image.

        Keyword arguments:
        img -- Original training image.

        Output:
        output -- resulting segmentation mask from algorithm.

        """

        # TODO We may want to move this? We need a number 1-4 smoothing iterations
        smoothing = int(self.params["alpha1"]*4)
        balloon = (self.params["alpha2"]*100)-50
        max_iter = self.params["max_iter"]
        level_set_shapes = ['checkerboard', 'circle']
        init_level_set = level_set_shapes[self.params['n_segments'] % 2]

        if(len(img.shape) > 2):
            if "channel" in self.params:
                channel = self.params['channel']
                img = img[:, :, channel]
            else:
                img = color.rgb2gray(img)

        # We run the inverse_gaussian_gradient to get the image to use
        gimage = skimage.segmentation.inverse_gaussian_gradient(
            img, self.params["beta1"], self.params["beta2"]
        )
        # zeros = 0
        output = skimage.segmentation.morphological_geodesic_active_contour(
            gimage,
            max_iter,
            init_level_set,
            smoothing,
            threshold="auto",
            balloon=balloon,
        )
        return output


segmentor.addsegmentor('MorphGeodesicActiveContour',
                       MorphGeodesicActiveContour)


##########################
