"""Segmentor library designed to learn how to segment images using GAs. This libary actually does not incode the GA itself, instead it just defines the search parameters the evaluation funtions and the fitness function (comming soon)."""
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
    """Run and evaluate the performance of an individual.

    Keyword arguments:
    img -- training image
    groundImg -- the ground truth for the image mask
    individual -- the list representing an individual in our population
    returnMask -- Boolean value indicating whether to return resulting mask for the individual or not (default False)

    Output:
    fitness -- resulting fitness value for the individual
    mask -- resulting image mask associated with the individual (if returnMask=True)

    """
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
    """Convert an individual's param list to an algorithm. Assumes order defined in the parameters class.

    Keyword arguments:
    individual -- the list representing an individual in our population

    Output:
    algorithm(individual) -- algorithm associated with the individual

    """
    if individual[0] in algorithmspace:
        algorithm = algorithmspace[individual[0]]
        return algorithm(individual)
    else:
        raise ValueError("Algorithm not avaliable")


class parameters(OrderedDict):
    """Construct an ordered dictionary that represents the search space.
    
    Functions:
    printparam -- returns description for each parameter
    tolist -- converts dictionary of params into list
    fromlist -- converts individual into dictionary of params

    """

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
        """Set default values for each param in the dictionary."""
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
            plist.append(self.params[key])
        return plist

    def fromlist(self, individual):
        """Convert individual's list into dictionary of params."""
        logging.getLogger().info(f"Parsing Parameter List for {len(individual)} parameters")
        for index, key in enumerate(self.pkeys):
            self[key] = individual[index]


class segmentor(object):
    """Base class for segmentor classes defined below.

    Functions:
    evaluate -- Run segmentation algorithm to get inferred mask.

    """

    algorithm = ""

    def __init__(self, paramlist=None):
        """Generate algorithm params from parameter list."""
        self.params = parameters()
        if paramlist:
            self.params.fromlist(paramlist)

    def evaluate(self, im):
        """Run segmentation algorithm to get inferred mask."""
        return np.zeros(im.shape[0:1])

    def __str__(self):
        """Return params for algorithm."""
        mystring = f"{self.params['algorithm']} -- \n"
        for p in self.paramindexes:
            mystring += f"\t{p} = {self.params[p]}\n"
        return mystring


class ColorThreshold(segmentor):
    """Peform Color Thresholding segmentation algorithm. Segments parts of the image based on the numerical values for the respective channel.

    Parameters:
    mx -- maximum thresholding value
    mn -- minimum thresholding value

    """

    def __init__(self, paramlist=None):
        """Get parameters from parameter list that are used in segmentation algorithm. Assign default values to these parameters."""
        super(ColorThreshold, self).__init__(paramlist)
        if not paramlist:
            self.params["algorithm"] = "CT"
            self.params["mu"] = 0.4
            self.params["sigma"] = 0.6
        self.paramindexes = ["sigma", "mu"]

    def evaluate(self, img): #XX
        """Evaluate segmentation algorithm on training image.

        Keyword arguments:
        img -- Original training image.

        Output:
        output -- resulting segmentation mask from algorithm.

        """
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

class Felzenszwalb(segmentor):
    """Perform Felzenszwalb segmentation algorithm. ONLY WORKS FOR RGB. The felzenszwalb algorithms computes a graph based on the segmentation. Produces an oversegmentation of the multichannel using min-span tree. Returns an integer mask indicating the segment labels.

    Parameters:
    scale -- float, higher meanse larger clusters
    sigma -- float, std. dev of Gaussian kernel for preprocessing
    min_size -- int, minimum component size. For postprocessing
    mulitchannel -- bool, Whether the image is 2D or 3D

    """

    def __doc__(self):
        """Return help string for function."""
        myhelp = "Wrapper function for the scikit-image Felzenszwalb segmentor:"
        myhelp += f" xx {skimage.segmentation.random_walker.__doc__}"
        return myhelp

    def __init__(self, paramlist=None):
        """Get parameters from parameter list that are used in segmentation algorithm. Assign default values to these parameters."""
        super(Felzenszwalb, self).__init__(paramlist)
        if not paramlist:
            self.params["algorithm"] = "FB"
            self.params["scale"] = 984
            self.params["sigma"] = 0.09
            self.params["min_size"] = 92
        self.paramindexes = ["scale", "sigma", "min_size"]

    def evaluate(self, img):
        """Evaluate segmentation algorithm on training image.

        Keyword arguments:
        img -- Original training image.

        Output:
        output -- resulting segmentation mask from algorithm.
        
        """
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
    """Perform the Slic segmentation algorithm. Segments k-means clustering in Color space (x, y, z). Returns a 2D or 3D array of labels.

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

    """

    def __init__(self, paramlist=None):
        """Get parameters from parameter list that are used in segmentation algorithm. Assign default values to these parameters."""
        super(Slic, self).__init__(paramlist)
        if not paramlist:
            self.params["algorithm"] = "SC"
            self.params["n_segments"] = 5
            self.params["compactness"] = 5
            self.params["iterations"] = 3
            self.params["sigma"] = 5
        self.paramindexes = ["n_segments", "compactness", "iterations", "sigma"]

    def evaluate(self, img):
        """Evaluate segmentation algorithm on training image.

        Keyword arguments:
        img -- Original training image.

        Output:
        output -- resulting segmentation mask from algorithm.
        
        """
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
    """Perform the Quick Shift segmentation algorithm. Segments images with quickshift clustering in Color (x,y) space. Returns ndarray segmentation mask of the labels.

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

    """

    def __init__(self, paramlist=None):
        """Get parameters from parameter list that are used in segmentation algorithm. Assign default values to these parameters."""
        super(QuickShift, self).__init__(paramlist)
        if not paramlist:
            self.params["algorithm"] = "QS"
            self.params["kernel_size"] = 5
            self.params["max_dist"] = 60
            self.params["sigma"] = 5
            self.params["seed"] = 1
        self.paramindexes = ["kernel_size", "max_dist", "sigma", "seed"]

    def evaluate(self, img):
        """Evaluate segmentation algorithm on training image.

        Keyword arguments:
        img -- Original training image.

        Output:
        output -- resulting segmentation mask from algorithm.
        
        """
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
    """Perform the Watershed segmentation algorithm. Uses user-markers. treats markers as basins and 'floods' them. Especially good if overlapping objects. Returns a labeled image ndarray.

    Parameters:
    image -- ndarray, input array
    compactness -- float, compactness of the basins. Higher values 
        make more regularly-shaped basin.

    """

    # Not using connectivity, markers, or offset params as arrays would
    # expand the search space too much.
    # abbreviation for algorithm = WS

    def __init__(self, paramlist=None):
        """Get parameters from parameter list that are used in segmentation algorithm. Assign default values to these parameters."""
        super(Watershed, self).__init__(paramlist)
        if not paramlist:
            self.params["algorithm"] = "WS"
            self.params["compactness"] = 2.0
        self.paramindexes = ["compactness"]

    def evaluate(self, img):
        """Evaluate segmentation algorithm on training image.

        Keyword arguments:
        img -- Original training image.

        Output:
        output -- resulting segmentation mask from algorithm.
        
        """
        output = skimage.segmentation.watershed(
            img, markers=None, compactness=self.params["compactness"]
        )
        return output


algorithmspace["WS"] = Watershed

class Chan_Vese(segmentor):
    """Peform Chan Vese segmentation algorithm. ONLY GRAYSCALE. Segments objects without clear boundaries. Returns segmentation array of algorithm.
    
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
        """Get parameters from parameter list that are used in segmentation algorithm. Assign default values to these parameters."""
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
        """Evaluate segmentation algorithm on training image.

        Keyword arguments:
        img -- Original training image.

        Output:
        output -- resulting segmentation mask from algorithm.
        
        """
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
    """Peform Morphological Chan Vese segmentation algorithm. ONLY WORKS ON GRAYSCALE. Active contours without edges. Can be used to segment images/volumes without good borders. Required that the inside of the object looks different than outside (color, shade, darker).
    
    Parameters:
    image -- ndarray of grayscale image
    iterations -- uint, number of iterations to run
    init_level_set -- str, or array same shape as image. Accepted string
        values are:
        'checkerboard': Uses checkerboard_level_set. Returns a binary level set of a checkerboard
        'circle': Uses circle_level_set. Creates a binary level set of a circle, given a radius and a
            center
    smoothing -- uint, number of times the smoothing operator is applied
        per iteration. Usually around 1-4. Larger values make it smoother
    lambda1 -- Weight param for outer region. If larger than lambda2, 
        outer region will give larger range of values than inner value.
    lambda2 -- Weight param for inner region. If larger thant lambda1, 
        inner region will have a larger range of values than outer region.

    """

    # Abbreviation for algorithm = MCV

    def __init__(self, paramlist=None):
        """Get parameters from parameter list that are used in segmentation algorithm. Assign default values to these parameters."""
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
        """Evaluate segmentation algorithm on training image.

        Keyword arguments:
        img -- Original training image.

        Output:
        output -- resulting segmentation mask from algorithm.
        
        """
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
    """Peform Morphological Geodesic Active Contour segmentation algorithm. Uses an image from inverse_gaussian_gradient in order to segment object with visible, but noisy/broken borders. inverse_gaussian_gradient computes the magnitude of the gradients in an image. Returns a preprocessed image suitable for above function. Returns ndarray of segmented image.

    Parameters: 
    gimage -- array, preprocessed image to be segmented.
    iterations -- uint, number of iterations to run.
    init_level_set -- str, array same shape as gimage. If string, possible
        values are:
        'checkerboard': Uses checkerboard_level_set. Returns a binary level set of a checkerboard
        'circle': Uses circle_level_set. Creates a binary level set of a circle, given a radius and a 
            center
    smoothing -- uint, number of times the smoothing operator is applied 
        per iteration. Usually 1-4, larger values have smoother segmentation.
    threshold -- Areas of image with a smaller value than the threshold are borders.
    balloon -- float, guides contour of low-information parts of image.

    """

    # Abbrevieation for algorithm = AC

    def __init__(self, paramlist=None):
        """Get parameters from parameter list that are used in segmentation algorithm. Assign default values to these parameters."""
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
        """Evaluate segmentation algorithm on training image.

        Keyword arguments:
        img -- Original training image.

        Output:
        output -- resulting segmentation mask from algorithm.
        
        """
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

def countMatches(inferred, groundTruth):
    """Map the segments in the inferred segmentation mask to the ground truth segmentation mask, and record the number of pixels in each of these mappings as well as the number of segments in both masks.
    
    Keyword arguments:
    inferred -- Resulting segmentation mask from individual.
    groundTruth -- Ground truth segmentation mask for training image.

    Outputs:
    setcounts -- Dictionary of dictionaries containing the number of pixels in 
        each segment mapping.
    len(m) -- Number of segments in inferred segmentation mask.
    len(n) -- Number of segments in ground truth segmentation mask.

    """
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

def countsets(setcounts):
    """For each inferred set, find the ground truth set which it maps the most pixels to. So we start from the inferred image, and map towards the ground truth image. For each i_key, the g_key that it maps the most pixels to is considered True. In order to see what ground truth sets have a corresponding set(s) in the inferred image, we record these "true" g_keys. This number of true g_keys is the value for L in our fitness function.

    Keyword arguments: 
    setcounts -- Dictionary of dictionaries containing the number of pixels in 
        each segment mapping.

    Outputs: 
    (total - p) -- Pixel error.
    L -- Number of ground truth segments that have a mapping in the inferred mask
    best -- True mapping as dictionary.

    """
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

def FitnessFunction(inferred, groundTruth):
    """Compute the fitness for an individual. Takes in two images and compares them according to the equation (p + 2)^log(|m - n| + 2), where p is the pixel error, m is the number of segments in the inferred mask, and n is the number of segments in the ground truth mask.

    Keyword arguments:
    inferred -- Resulting segmentation mask from individual.
    groundTruth -- Ground truth segmentation mask for training image.

    Outputs:
    error -- fitness value as float
    best -- true mapping as dictionary

    """
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
