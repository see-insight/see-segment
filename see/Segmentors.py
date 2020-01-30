""" Segmentor library designed to learn how to segment images using GAs.
This libary actually does not incode the GA itself, instead it just defines
the search parameters the evaluation funtions and the fitness function (comming soon)
"""

# TODO: Research project-clean up the parameters class to reduce the search space
# TODO: Change the seed from a number to a fraction 0-1 which is scaled to image rows and columns
# TODO: Enumerate teh word based measures.

from collections import OrderedDict
import sys

import numpy as np
import skimage
from skimage import segmentation
from skimage import color
from PIL import Image
import pandas as pd  # used in fitness? Can it be removed?
import logging

# List of all algorithms
algorithmspace = dict()


def runAlgo(img, groundImg, individual, returnMask=False):
    logging.getLogger().info(f"Running Algorithm {individual[0]}")
    # img = copy.deepcopy(copyImg)
    seg = algoFromParams(individual)
    mask = seg.evaluate(img)
    logging.getLogger().info("Calculating Fitness")
    fitness = FitnessFunction(mask, groundImg)
    if returnMask:
        return [fitness, mask]
    else:
        return fitness


def algoFromParams(individual):
    """Converts a param list to an algorithm Assumes order 
    defined in the parameters class"""
    if individual[0] in algorithmspace:
        algorithm = algorithmspace[individual[0]]
        return algorithm(individual)
    else:
        raise ValueError("Algorithm not avaliable")


class parameters(OrderedDict):
    descriptions = dict()
    ranges = dict()
    pkeys = []

    ranges["algorithm"] = "['CT','FB','SC','WS','CV','MCV','AC']"
    descriptions["algorithm"] = "string code for the algorithm"

    descriptions["beta"] = "A parameter for randomWalker So, I should take this out"
    ranges["beta"] = "[i for i in range(0,10000)]"

    descriptions["tolerance"] = "A parameter for flood and flood_fill"
    ranges["tolerance"] = "[float(i)/1000 for i in range(0,1000,1)]"

    descriptions["scale"] = "A parameter for felzenszwalb"
    ranges["scale"] = "[i for i in range(0,10000)]"

    descriptions["sigma"] = "sigma value. A parameter for felzenswalb, inverse_guassian_gradient, slic, and quickshift"
    ranges["sigma"] = "[float(i)/100 for i in range(0,10,1)]"

    descriptions["min_size"] = "parameter for felzenszwalb"
    ranges["min_size"] = "[i for i in range(0,10000)]"

    descriptions["n_segments"] = "A parameter for slic"
    ranges["n_segments"] = "[i for i in range(2,10000)]"

    descriptions["iterations"] = "A parameter for both morphological algorithms"
    ranges["iterations"] = "[10, 10]"

    descriptions["ratio"] = "A parameter for ratio"
    ranges["ratio"] = "[float(i)/100 for i in range(0,100)]"

    descriptions["kernel_size"] = "A parameter for kernel_size"
    ranges["kernel_size"] = "[i for i in range(0,10000)]"

    descriptions["max_dist"] = "A parameter for quickshift"
    ranges["max_dist"] = "[i for i in range(0,10000)]"

    descriptions["seed"] = "A parameter for quickshift, and perhaps other random stuff"
    ranges["seed"] = "[134]"

    descriptions["connectivity"] = "A parameter for flood and floodfill"
    ranges["connectivity"] = "[i for i in range(0, 9)]"

    descriptions["compactness"] = "A parameter for slic and watershed"
    ranges["compactness"] = "[0.0001,0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]"

    descriptions["mu"] = "A parameter for chan_vese"
    ranges["mu"] = "[float(i)/100 for i in range(0,100)]"

    descriptions["lambda"] = "A parameter for chan_vese and morphological_chan_vese"
    ranges["lambda"] = "[(1,1), (1,2), (2,1)]"

    descriptions["dt"] = "#An algorithm for chan_vese May want to make seperate level sets for different functions e.g. Morph_chan_vese vs morph_geo_active_contour"
    ranges["dt"] = "[float(i)/10 for i in range(0,100)]"

    descriptions["init_level_set_chan"] = "A parameter for chan_vese and morphological_chan_vese"
    ranges["init_level_set_chan"] = "['checkerboard', 'disk', 'small disk']"

    descriptions["init_level_set_morph"] = "A parameter for morphological_chan_vese"
    ranges["init_level_set_morph"] = "['checkerboard', 'circle']"

    descriptions["smoothing"] = "A parameter used in morphological_geodesic_active_contour"
    ranges["smoothing"] = "[i for i in range(1, 10)]"

    descriptions["alpha"] = "A parameter for inverse_guassian_gradient"
    ranges["alpha"] = "[i for i in range(0,10000)]"

    descriptions["balloon"] = "A parameter for morphological_geodesic_active_contour"
    ranges["balloon"] = "[i for i in range(-50,50)]"

    descriptions["seed_pointX"] = "A parameter for flood and flood_fill"
    ranges["seed_pointX"] = "[0.0]"

    descriptions["seed_pointY"] = "??"
    ranges["seed_pointY"] = "[0.0]"

    descriptions["seed_pointZ"] = "??"
    ranges["seed_pointZ"] = "[0.0]"

    #     Try to set defaults only once.
    #     Current method may cause all kinds of weird problems.
    #     @staticmethod
    #     def __Set_Defaults__()

    def __init__(self):
        self["algorithm"] = "None"
        self["beta"] = 0.0
        self["tolerance"] = 0.0
        self["scale"] = 0.0
        self["sigma"] = 0.0
        self["min_size"] = 0.0
        self["n_segments"] = 0.0
        self["iterations"] = 10
        self["ratio"] = 0.0
        self["kernel_size"] = 0.0
        self["max_dist"] = 0.0
        self["seed"] = 0.0
        self["connectivity"] = 0.0
        self["compactness"] = 0.0
        self["mu"] = 0.0
        self["lambda"] = (1, 1)
        self["dt"] = 0.0
        self["init_level_set_chan"] = "disk"
        self["init_level_set_morph"] = "checkerboard"
        self["smoothing"] = 0.0
        self["alpha"] = 0.0
        self["balloon"] = 0.0
        self["seed_pointX"] = 0.0
        self["seed_pointY"] = 0.0
        self["seed_pointZ"] = 0.0
        self.pkeys = list(self.keys())

    def printparam(self, key):
        return f"{key}={self[key]}\n\t{self.descriptions[key]}\n\t{self.ranges[key]}\n"

    def __str__(self):
        out = ""
        for index, k in enumerate(self.pkeys):
            out += f"{index} " + self.printparam(k)
        return out

    def tolist(self):
        plist = []
        for key in self.pkeys:
            plist.append(self.params[key])
        return plist

    def fromlist(self, individual):
        logging.getLogger().info(f"Parsing Parameter List for {len(individual)} parameters")
        for index, key in enumerate(self.pkeys):
            self[key] = individual[index]


class segmentor(object):
    algorithm = ""

    def __init__(self, paramlist=None):
        self.params = parameters()
        if paramlist:
            self.params.fromlist(paramlist)

    def evaluate(self, im):
        return np.zeros(im.shape[0:1])

    def __str__(self):
        mystring = f"{self.params['algorithm']} -- \n"
        for p in self.paramindexes:
            mystring += f"\t{p} = {self.params[p]}\n"
        return mystring


class ColorThreshold(segmentor):
    def __init__(self, paramlist=None):
        super(ColorThreshold, self).__init__(paramlist)
        if not paramlist:
            self.params["algorithm"] = "CT"
            self.params["mu"] = 0.4
            self.params["sigma"] = 0.6
        self.paramindexes = ["sigma", "mu"]

    def evaluate(self, img): #XX
        channel_num = 1  # TODO: Need to make this a searchable parameter.
        if len(img.shape) > 2:
            if channel_num < img.shape[2]:
                channel = img[:, :, channel_num]
            else:
                channel = img[:, :, 0]
        else:
            channel = img
        pscale = np.max(channel)
        mx = self.params["sigma"] * pscale
        mn = self.params["mu"] * pscale
        if mx < mn:
            temp = mx
            mx = mn
            mn = temp

        output = np.ones(channel.shape)
        output[channel < mn] = 0
        output[channel > mx] = 0

        return output


algorithmspace["CT"] = ColorThreshold

"""
#felzenszwalb
#ONLY WORKS FOR RGB
https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_segmentations.html
The felzenszwalb algorithms computes a graph based on the segmentation
Produces an oversegmentation of the multichannel using min-span tree.
Returns an integer mask indicating the segment labels

#Variables
scale: float, higher meanse larger clusters
sigma: float, std. dev of Gaussian kernel for preprocessing
min_size: int, minimum component size. For postprocessing
mulitchannel: bool, Whether the image is 2D or 3D. 2D images
are not supported at all
"""
class Felzenszwalb(segmentor):
    def __doc__(self):
        myhelp = "Wrapper function for the scikit-image Felzenszwalb segmentor:"
        myhelp += f" xx {skimage.segmentation.random_walker.__doc__}"
        return myhelp

    def __init__(self, paramlist=None):
        super(Felzenszwalb, self).__init__(paramlist)
        if not paramlist:
            self.params["algorithm"] = "FB"
            self.params["scale"] = 984
            self.params["sigma"] = 0.09
            self.params["min_size"] = 92
        self.paramindexes = ["scale", "sigma", "min_size"]

    def evaluate(self, img):
        multichannel = False
        if len(img.shape) > 2:
            multichannel = True
        output = skimage.segmentation.felzenszwalb(
            img,
            self.params["scale"],
            self.params["sigma"],
            self.params["min_size"],
            multichannel=True,
        )
        return output


algorithmspace["FB"] = Felzenszwalb


class Slic(segmentor):
    """
    #slic
    https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_segmentations.html
    segments k-means clustering in Color space (x, y, z)
    #Returns a 2D or 3D array of labels

    #Variables
    image -- ndarray, input image
    n_segments -- int,  number of labels in segmented output image 
        (approx). Should find a way to compute n_segments
    compactness -- float, Balances color proximity and space proximity.
        Higher values mean more weight to space proximity (superpixels
        become more square/cubic) #Recommended log scale values (0.01, 
        0.1, 1, 10, 100, etc)
    max_iter -- int, max number of iterations of k-means
    sigma -- float or (3,) shape array of floats,  width of Guassian
        smoothing kernel. For pre-processing for each dimesion of the
        image. Zero means no smoothing
    spacing -- (3,) shape float array : voxel spacing along each image
        dimension. Defalt is uniform spacing
    multichannel -- bool,  multichannel (True) vs grayscale (False)
    #Needs testing to find correct values

    #Abbreviation for algorithm == SC
    """

    def __init__(self, paramlist=None):
        super(Slic, self).__init__(paramlist)
        if not paramlist:
            self.params["algorithm"] = "SC"
            self.params["n_segments"] = 5
            self.params["compactness"] = 5
            self.params["iterations"] = 3
            self.params["sigma"] = 5
        self.paramindexes = ["n_segments", "compactness", "iterations", "sigma"]

    def evaluate(self, img):
        multichannel = False
        if len(img.shape) > 2:
            multichannel = True
        output = skimage.segmentation.slic(
            img,
            n_segments=self.params["n_segments"],
            compactness=self.params["compactness"],
            max_iter=self.params["iterations"],
            sigma=self.params["sigma"],
            convert2lab=True,
            multichannel=multichannel,
        )
        return output


algorithmspace["SC"] = Slic


class QuickShift(segmentor):
    """
    #quickshift
    https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_segmentations.html
    Segments images with quickshift clustering in Color (x,y) space
    #Returns ndarray segmentation mask of the labels
    #Variables
    image -- ndarray, input image
    ratio -- float, balances color-space proximity & image-space  proximity. Higher vals give more weight to color-space
    kernel_size: float, Width of Guassian kernel using smoothing. Higher means fewer clusters
    max_dist -- float: Cut-off point for data distances. Higher means fewer clusters
    return_tree -- bool: Whether to return the full segmentation hierachy tree and distances. Set as False
    sigma -- float: Width of Guassian smoothing as preprocessing.Zero means no smoothing
    convert2lab -- bool: leave alone
    random_seed -- int, Random seed used for breacking ties. 
    """

    def __init__(self, paramlist=None):
        super(QuickShift, self).__init__(paramlist)
        if not paramlist:
            self.params["algorithm"] = "QS"
            self.params["kernel_size"] = 5
            self.params["max_dist"] = 60
            self.params["sigma"] = 5
            self.params["seed"] = 1
        self.paramindexes = ["kernel_size", "max_dist", "sigma", "seed"]

    def evaluate(self, img):
        output = skimage.segmentation.quickshift(
            img,
            ratio=self.params["ratio"],
            kernel_size=self.params["kernel_size"],
            max_dist=self.params["max_dist"],
            sigma=self.params["sigma"],
            random_seed=self.params["seed"],
        )
        return output


algorithmspace["QS"] = QuickShift


class Watershed(segmentor):
    """
    #Watershed
    https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_watershed.html
    Uses user-markers. treats markers as basins and 'floods' them.
    Especially good if overlapping objects. 
    #Returns a labeled image ndarray
    #Variables
    image -> ndarray, input array
    markers -> int, or int ndarray same shape as image: markers indicating 'basins'
    connectivity -> ndarray, indicates neighbors for connection
    offset -> array, same shape as image: offset of the connectivity
    mask -> ndarray of bools (or 0s and 1s): 
    compactness -> float, compactness of the basins Higher values make more regularly-shaped basin
    """

    # Not using connectivity, markers, or offset params as arrays would
    # expand the search space too much.
    # abbreviation for algorithm = WS

    def __init__(self, paramlist=None):
        super(Watershed, self).__init__(paramlist)
        if not paramlist:
            self.params["algorithm"] = "WS"
            self.params["compactness"] = 2.0
        self.paramindexes = ["compactness"]

    def evaluate(self, img):
        output = skimage.segmentation.watershed(
            img, markers=None, compactness=self.params["compactness"]
        )
        return output


algorithmspace["WS"] = Watershed


class Chan_Vese(segmentor):
    """
    #chan_vese
    #ONLY GRAYSCALE
    https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_chan_vese.html
    Segments objects without clear boundaries
    #Returns: segmentation array of algorithm. Optional: When the algorithm converges
    #Variables
    image -> ndarray grayscale image to be segmented
    mu -> float, 'edge length' weight parameter. Higher mu vals make a 'round edge' closer to zero will detect smaller objects
    lambda1 -> float 'diff from average' weight param to determine if 
        output region is True. If lower than lambda1, the region has a 
        larger range of values than the other
    lambda2 -> float 'diff from average' weight param to determine if 
        output region is False. If lower than lambda1, the region will 
        have a larger range of values
    Note: Typical values for mu are from 0-1. 
    Note: Typical values for lambda1 & lambda2 are 1. If the background 
        is 'very' different from the segmented values, in terms of
        distribution, then the lambdas should be different from 
        eachother
    tol: positive float, typically (0-1), very low level set variation 
        tolerance between iterations.
    max_iter: uint,  max number of iterations before algorithms stops
    dt: float, Multiplication factor applied at the calculations step
    init_level_set: str/ndarray, defines starting level set used by
        algorithm. Accepted values are:
        'checkerboard': fast convergence, hard to find implicit edges
        'disk': Somewhat slower convergence, more likely to find
            implicit edges
        'small disk': Slowest convergence, more likely to find implicit edges
        can also be ndarray same shape as image
    extended_output: bool, If true, adds more returns 
    (Final level set & energies)
    """

    # Abbreviation for Algorithm = CV

    def __init__(self, paramlist=None):
        super(Chan_Vese, self).__init__(paramlist)
        if not paramlist:
            self.params["algorithm"] = "CV"
            self.params["mu"] = 2.0
            self.params["lambda"] = (10, 20)
            self.params["iterations"] = 10
            self.params["dt"] = 0.10
            self.params["init_level_set_chan"] = "small disk"
        self.paramindexes = ["mu", "lambda", "iterations", "dt", "init_level_set_chan"]

    def evaluate(self, img):
        if len(img.shape) == 3:
            img = skimage.color.rgb2gray(img)
        output = skimage.segmentation.chan_vese(
            img,
            mu=self.params["mu"],
            lambda1=self.params["lambda"][0],
            lambda2=self.params["lambda"][1],
            tol=self.params["tolerance"],
            max_iter=self.params["iterations"],
            dt=self.params["dt"],
        )
        return output


algorithmspace["CV"] = Chan_Vese


class Morphological_Chan_Vese(segmentor):
    """
    #morphological_chan_vese
    #ONLY WORKS ON GRAYSCALE
    https://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.morphological_chan_vese
    Active contours without edges. Can be used to segment images/
        volumes without good borders. Required that the inside of the
        object looks different than outside (color, shade, darker).
    #Returns Final segmention
    #Variables:
    image -> ndarray of grayscale image
    iterations -> uint, number of iterations to run
    init_level_set: str, or array same shape as image. Accepted string
        values are:
        'checkerboard': Uses checkerboard_level_set
        https://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.checkerboard_level_set
        returns a binary level set of a checkerboard
        'circle': Uses circle_level_set
        https://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.circle_level_set
        Creates a binary level set of a circle, given a radius and a
        center

    smoothing: uint, number of times the smoothing operator is applied
        per iteration. Usually around 1-4. Larger values make stuf 
        smoother
    lambda1: Weight param for outer region. If larger than lambda2, 
        outer region will give larger range of values than inner value
    lambda2: Weight param for inner region. If larger thant lambda1, 
        inner region will have a larger range of values than outer region
    """

    # Abbreviation for algorithm = MCV

    def __init__(self, paramlist=None):
        super(Morphological_Chan_Vese, self).__init__(paramlist)
        if not paramlist:
            self.params["algorithm"] = "MCV"
            self.params["iterations"] = 10
            self.params["init_level_set_morph"] = "checkerboard"
            self.params["smoothing"] = 10
            self.params["lambda"] = (10, 20)
        self.paramindexes = [
            "iterations",
            "init_level_set_morph",
            "smoothing",
            "lambda",
        ]

    def evaluate(self, img):
        if len(img.shape) == 3:
            img = skimage.color.rgb2gray(img)
        output = skimage.segmentation.morphological_chan_vese(
            img,
            iterations=self.params["iterations"],
            init_level_set=self.params["init_level_set_morph"],
            smoothing=self.params["smoothing"],
            lambda1=self.params["lambda"][0],
            lambda2=self.params["lambda"][1],
        )
        return output


algorithmspace["MCV"] = Morphological_Chan_Vese


class MorphGeodesicActiveContour(segmentor):
    """
    #morphological_geodesic_active_contour
    https://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.morphological_geodesic_active_contour
    Uses an image from inverse_gaussian_gradient in order to segment
        object with visible, but noisy/broken borders
    #inverse_gaussian_gradient
    https://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.inverse_gaussian_gradient
    Compute the magnitude of the gradients in an image. returns a
        preprocessed image suitable for above function
    #Returns ndarray of segmented image
    #Variables
    gimage: array, preprocessed image to be segmented
    iterations: uint, number of iterations to run
    init_level_set: str, array same shape as gimage. If string, possible
        values are:
        'checkerboard': Uses checkerboard_level_set
        https://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.checkerboard_level_set
        returns a binary level set of a checkerboard
        'circle': Uses circle_level_set
        https://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.circle_level_set
        Creates a binary level set of a circle, given a radius and a 
        center
    smoothing: uint, number of times the smoothing operator is applied 
        per iteration. Usually 1-4, larger values have smoother 
        segmentation
    threshold: Areas of image with a smaller value than the threshold
        are borders
    balloon: float, guides contour of low-information parts of image, 	
    """

    # Abbrevieation for algorithm = AC

    def __init__(self, paramlist=None):
        super(MorphGeodesicActiveContour, self).__init__(paramlist)
        if not paramlist:
            self.params["algorithm"] = "AC"
            self.params["alpha"] = 0.2
            self.params["sigma"] = 0.3
            self.params["iterations"] = 10
            self.params["init_level_set_morph"] = "checkerboard"
            self.params["smoothing"] = 5
            self.params["balloon"] = 10
        self.paramindexes = [
            "alpha",
            "sigma",
            "iterations",
            "init_level_set_morph",
            "smoothing",
            "balloon",
        ]

    def evaluate(self, img):
        # We run the inverse_gaussian_gradient to get the image to use
        gimage = skimage.segmentation.inverse_gaussian_gradient(
            img, self.params["alpha"], self.params["sigma"]
        )
        zeros = 0
        output = skimage.segmentation.morphological_geodesic_active_contour(
            gimage,
            self.params["iterations"],
            self.params["init_level_set_morph"],
            smoothing=self.params["smoothing"],
            threshold="auto",
            balloon=self.params["balloon"],
        )
        return output


algorithmspace["AC"] = MorphGeodesicActiveContour

# class Flood(segmentor):
#     '''
#     #flood
#     #DOES NOT SUPPORT MULTICHANNEL IMAGES
#     https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_floodfill.html
#     Uses a seed point and to fill all connected points within/equal to
#         a tolerance around the seed point
#     #Returns a boolean array with 'flooded' areas being true
#     #Variables
#     image: ndarray, input image
#     seed_point: tuple/int, x,y,z referring to starting point for flood
#         fill
#     selem: ndarray of 1's and 0's, Used to determine neighborhood of
#         each pixel
#     connectivity: int, Used to find neighborhood of each pixel. Can use
#         this or selem.
#     tolerance: float or int, If none, adjacent values must be equal to
#         seed_point. Otherwise, how likely adjacent values are flooded.
#     '''
#     #Abbreviation for algorithm = FD

#     def __init__(self, paramlist=None):
#         super(Flood, self).__init__(paramlist)
#         self.params['algorithm'] = 'AC'
#         self.params['seed_pointX'] = 10
#         self.params['seed_pointY'] = 20
#         self.params['seed_pointZ'] = 0
#         self.params['connect'] = 4
#         self.params['tolerance'] = 0.5
#         self.paramindexes = ['seed', 'connect', 'tolerance']

#     def evaluate(self, img):
#         output = skimage.segmentation.flood(
#             img,
#             (self.params['seed_pointX'],
#              self.params['seed_pointY'],
#              self.params['seed_pointZ']),
#             connectivity=self.params['connect'],
#             tolerance=self.params['tolerance'])
#         return output
# algorithmspace['FD'] = Flood


# class FloodFill(segmentor):
#     '''
#     #flood_fill
#     https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_floodfill.html
#     Like a paint-bucket tool in paint. Like flood, but changes the
#         color equal to new_type
#     #Returns A filled array of same shape as the image
#     #Variables
#     image: ndarray, input image
#     seed_point: tuple or int, starting point for filling (x,y,z)
#     new_value: new value to set the fill to (e.g. color). Must agree
#         with image type
#     selem: ndarray, Used to find neighborhood of filling
#     connectivity: Also used to find neighborhood of filling if selem is
#         None
#     tolerance: float or int, If none, adjacent values must be equal to
#         seed_point. Otherwise, how likely adjacent values are flooded.
#     inplace: bool, If true, the flood filling is applied to the image,
#         if False, the image is not modified. Default False, don't
#         change
#     '''
#     #Abbreviation for algorithm == FF

#     def __init__(self, paramlist=None):
#         super(FloodFill, self).__init__(paramlist)
#         self.params['algorithm'] = 'AC'
#         self.params['seed_pointX'] = 10
#         self.params['seed_pointY'] = 20
#         self.params['seed_pointZ'] = 0
#         self.params['connect'] = 4
#         self.params['tolerance'] = 0.5
#         self.paramindexes = ['seed', 'connect', 'tolerance']

#     def evaluate(self, img):
#         output = skimage.segmentation.flood_fill(
#             img,
#             (self.params['seed_pointX'],
#              self.params['seed_pointY'],
#              self.params['seed_pointZ']),
#             134,  #TODO: Had coded value
#             connectivity= self.params['connect'],
#             tolerance=self.params['tolerance'])
#         try:
#             #I'm not sure if this will work on grayscale
#             image = Image.fromarray(output.astype('uint8'), '1')
#         except ValueError:
#             image = Image.fromarray(output.astype('uint8'), 'RGB')

#         width = image.width
#         height = image.width


#         #Converting the background to black
#         for x in range(0, width):
#             for y in range(0, height):
#                 #First check for grayscale
#                 pixel = image.getpixel((x,y))
#                 if pixel[0] == 134:
#                     image.putpixel((x,y), 134)
#                     continue
#                 else:
#                     image.putpixel((x,y), 0)
#                     #print(image.getpixel((x,y)))

#         #image.convert(mode='L')
#         pic = np.array(image)
#         return pic
# algorithmspace['FF'] = FloodFill

# TODO: Figure out the mask part?
# class RandomWalker(segmentor):
#     algorithm = 'RW'
#     paramindexes = [1, 2]

#     def __doc__(self):
#         myhelp = "Wrapper function for the scikit-image random_walker segmentor:"
#         myhelp += f" xx {skimage.segmentation.random_walker.__doc__}"
#         return myhelp

#     def __init__(self, beta = 0.5, tolerance = 0.4):
#         self.beta = beta
#         self.tolerance = tolerance

#     def evaluate(self, img):
#         #Let's deterime what mode to use
#         mode = "bf"
#         if len(img) < 512 :
#             mode = "cg_mg"

#         #If data is 2D, then this is a grayscale, so multichannel is
#         output = skimage.segmentation.random_walker(
#             img, labels=mask,
#             beta=self.beta,
#             tol=self.tolerance, copy=True,
#             multichannel=True, return_full_prob=False)
#         return output



"""
function to calculate number of sets in our test image
that map to more than one set in our truth image, and how many
pixels are in those sets. Used in fitness function below.
INPUTS: truth image, infer image
RETURNS: number of repeated sets, number of pixels in repeated sets
"""
def set_fitness_func(a_test, b_test, include_L=False):
    
    #TODO: This is redundant. We just pass in the raveled vector from fitnessfunciton.
    a_test_int = a_test.ravel().astype(int)  # turn float array into int array
    b_test_int = b_test.ravel().astype(int)  # turn float array into in array

    assert len(a_test_int == len(b_test_int))
    
    # create char array to separate two images
    filler = np.chararray((len(a_test_int)))
    filler[:] = ":"

    # match arrays so we can easily compare
    matched = np.core.defchararray.add(a_test_int.astype(str), filler.astype(str))
    matched = np.core.defchararray.add(matched, b_test_int.astype(str))

    # collect unique set pairings
    unique_sets = np.unique(matched)

    # count number of pixels for each set pairing
    set_counts = {}
    for i in unique_sets:
        set_counts[i] = sum(matched[:] == i)#sum(np.core.defchararray.count(matched, i))

    # print statements for debugging
    #     print('UNIQUE: ', unique_sets) # see set pairings
    #     print('SET_COUNTS: ', set_counts) # see counts

    # counts every repeated set. EX: if we have (A, A, B, B, B, C) we get 5 repeated.
    sets = set()  # init container that will hold all sets in infer. image
    repeats = []  # init container that will hold all repeated sets
    b_set_counts = (
        {}
    )  # init container that will hold pixel counts for each repeated set
    for i in unique_sets:
        current_set = i[i.find(":") + 1 :]  # get inf. set from each pairing
        if current_set in sets:  # if repeat set
            repeats.append(current_set)  # add set to repeats list
            # add pixel count to set in dict.
            b_set_counts[current_set].append(set_counts[i])
        elif current_set not in sets:  # if new set
            # init. key and add pixel count
            b_set_counts[current_set] = [set_counts[i]]
            sets.add(current_set)  # add set to sets container

    # get number of repeated sets
    num_repeats = len(np.unique(repeats)) + len(repeats)
    # num_repeats = len(sets)## get all sets in infer image

    # count number of pixels in all repeated sets. Assumes pairing with max. num
    # of pixels is not error
    repeat_count = 0
    used_sets = set()
    for i in b_set_counts.keys():
        repeat_count += sum(b_set_counts[i]) - max(b_set_counts[i])
        for j in unique_sets:
            if j[j.find(":") + 1 :] == i and set_counts[j] == max(b_set_counts[i]):
                used_sets.add(j[: j.find(":")])

    if include_L == True:
        return num_repeats, repeat_count, used_sets
    else:
        return num_repeats, repeat_count


"""Takes in two ImageData obects and compares them according to
skimage's Structual Similarity Index and the mean squared error
Variables:
img1 is an image array segmented by the algorithm.
img2 is the validation image
imgDim is the number of dimensions of the image.
"""


def FitnessFunction_old(mask1, mask2):
    # assert(len(img1.shape) == len(img2.shape) == imgDim)

    # #The channel deterimines if this is a RGB or grayscale image
    # channel = False
    # if imgDim > 2: channel = True
    # #print(img1.dtype, img2.dtype)
    # img1 = np.uint8(img1)
    # #print(img1.dtype, img2.dtype)
    # assert(img1.dtype == img2.dtype)
    # #TODO: Change to MSE
    # #Comparing the Structual Similarity Index (SSIM) of two images
    # ssim = skimage.measure.compare_ssim(img1, img2, win_size=3,
    #    multichannel=channel, gaussian_weights=True)
    # #Comparing the Mean Squared Error of the two image
    # #print("About to compare")
    # #print(img1.shape, img2.shape, imgDim)
    # #mse = skimage.measure.compare_mse(img1, img2)
    # #Deleting the references to the objects and freeing memory
    # del img1
    # del img2
    # #print("eror above?")
    # return [abs(ssim),]

    # makes sure images are in grayscale
    if len(mask1.shape) > 2:
        llogging.getLogger().info("mask1 not in grayscale")
        mask1 = color.rgb2gray(mask1)
    if len(mask2.shape) > 2:  # comment out
        logging.getLogger().info("img2 not in grayscale")
        mask2 = color.rgb2gray(mask2)  # comment out
    # img2 = img2[:,:,0]#color.rgb2gray(true_im) # convert to grayscale
    # img2[img2[:,:] != 0] = 1
    
    # makes sure images can be read as segmentation labels (i.e. integers)
    mask1 = pd.factorize(mask1.ravel())[0].reshape(mask1.shape)
    mask2 = pd.factorize(mask2.ravel())[0].reshape(mask2.shape)  # comment out

    # Replace with function to output p an L
    # p - number of pixels not correcly mapped
    # L - Number of correctly mapped sets
    num_repeats, p, used_sets = set_fitness_func(mask2, mask1, True)
    
    m = len(np.unique(mask1)) # Number of unique labels in mask1
    n = len(np.unique(mask2)) # Number of unique labels in mask1
    L = len(used_sets) # number of true sets (i.e. used)
    
    logging.getLogger().info(f"p={p}, m={m}, n={n}, L={L}")
    
    error = (p + 2) ** np.log(abs(m - n) + 2)  # / (L >= n)
    print(f"TESTING - L={L} < n={n} p={p} m={m} error = {error} ")
    # error = (repeat_count + 2)**(abs(m - n)+1)
    if (L < n) or error <= 0 or error == np.inf or error == np.nan:
        print(
            f"WARNING: Fitness bounds exceeded, using Maxsize - {L} < {n} or {error} <= 0 or {error} == np.inf or {error} == np.nan:"
        )
        error = sys.maxsize
        # print(error)
    return [
        error,
    ]

def countMatches(inferred, groundTruth):
    assert (inferred.shape == groundTruth.shape)    
    m = set()
    n = set()
    setcounts = dict()
    for r in range(inferred.shape[0]):
        for c in range(inferred.shape[1]):
            i_key = inferred[r,c]
            m.add(i_key)
            g_key = groundTruth[r,c]
            n.add(g_key)
            if i_key in setcounts:
                if g_key in setcounts[i_key]:
                    setcounts[i_key][g_key] += 1
                else:
                    setcounts[i_key][g_key] = 1
            else:
                setcounts[i_key] = dict()
                setcounts[i_key][g_key] = 1
    return setcounts, len(m), len(n)

'''
For each inferred set, find the ground truth set which it maps the most 
pixels to. So we start from the inferred image, and map towards the 
ground truth image. For each i_key, the g_key that it maps the most 
pixels to is considered True. In order to see what ground truth sets
have a corresponding set(s) in the inferred image, we record these "true" g_keys. 
This number of true g_keys is the value for L in our fitness function.
'''
def countsets(setcounts):
    p = 0
    #L = len(setcounts)
    
    total = 0
    Lsets = set()
    
    best = dict()
    
    for i_key in setcounts: 
        mx = 0
        mx_key = ''
        for g_key in setcounts[i_key]:
            total += setcounts[i_key][g_key] # add to total pixel count
            if setcounts[i_key][g_key] > mx:
                mx = setcounts[i_key][g_key]
                # mx_key = i_key
                mx_key = g_key # record mapping with greatest pixel count
        p += mx
        # Lsets.add(g_key)
        Lsets.add(mx_key) # add the g_key we consider to be correct
        # best[i_key] = g_key
        best[i_key] = mx_key # record "true" mapping
    L = len(Lsets)
    return total-p,L, best


"""Takes in two ImageData obects and compares them according to
skimage's Structual Similarity Index and the mean squared error
Variables:
img1 is the validation image
img2 is an image array segmented by the algorithm.
imgDim is the number of dimensions of the image.
"""
def FitnessFunction(inferred, groundTruth):
    # makes sure images are in grayscale
    if len(inferred.shape) > 2:
        logging.getLogger().info("inferred not in grayscale")
        inferred = color.rgb2gray(inferred)
    if len(groundTruth.shape) > 2:  # comment out
        logging.getLogger().info("img2 not in grayscale")
        groundTruth = color.rgb2gray(groundTruth)  # comment out
    
    # Replace with function to output p an L
    # p - number of pixels not correcly mapped
    # L - Number of correctly mapped sets
    setcounts, m, n = countMatches(inferred, groundTruth)
    
    #print(setcounts)
    p, L, best = countsets(setcounts)
    
    logging.getLogger().info(f"p={p}, m={m}, n={n}, L={L}")
    
    error = (p + 2) ** np.log(abs(m - n) + 2)  # / (L >= n)
    # error = (repeat_count + 2)**(abs(m - n)+1)
    print(f"TESTING - L={L} < n={n} p={p} m={m} error = {error} ")
    if (L < n) or error <= 0 or error == np.inf or error == np.nan:
        logging.warning(
            f"WARNING: Fitness bounds exceeded, using Maxsize - {L} < {n} or {error} <= 0 or {error} == np.inf or {error} == np.nan:"
        )
        error = sys.maxsize
        # print(error)
    return [error, best]
