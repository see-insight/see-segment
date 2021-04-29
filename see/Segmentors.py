"""Segmentor library designed to learn how to segment images using GAs.
 This libary actually does not incode the GA itself, instead it just defines
  the search parameters the evaluation funtions and the fitness function (comming soon)."""
# DO: Research project-clean up the parameters class to reduce the search space
# DO: Change the seed from a number to a fraction 0-1 which is scaled to image rows and columns
# DO: Enumerate teh word based measures.
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

# List of all algorithms
algorithmspace = dict()

def mutateAlgo(copy_child, pos_vals, flip_prob=0.5, seed=False):
    """Generate an offspring based on current individual."""

    child = copy.deepcopy(copy_child)
    
    # Not every algorithm is associated with every value
    # Let's first see if we change the algorithm
    rand_val = random.random()
    if rand_val < flip_prob:
        # Let's mutate the algorithm
        child[0] = random.choice(pos_vals[0])

    #use the local search for mutation.
    seg = algoFromParams(child)
    child = seg.mutateself(flip_prob)
    return child


def runAlgo(img, ground_img, individual, return_mask=False):
    """Run and evaluate the performance of an individual.

    Keyword arguments:
    img -- training image
    ground_img -- the ground truth for the image mask
    individual -- the list representing an individual in our population
    return_mask -- Boolean value indicating whether to return resulting
     mask for the individual or not (default False)

    Output:
    fitness -- resulting fitness value for the individual
    mask -- resulting image mask associated with the individual (if return_mask=True)

    """
    logging.getLogger().info(f"Running Algorithm {individual[0]}")
    # img = copy.deepcopy(copyImg)
    seg = algoFromParams(individual)
    mask = seg.evaluate(img)
    logging.getLogger().info("Calculating Fitness")
    fitness = FitnessFunction(mask, ground_img)
    if return_mask:
        print("Returning mask")
        return [fitness, mask]
    else:
        return fitness


def algoFromParams(individual):
    """Convert an individual's param list to an algorithm. Assumes order
     defined in the parameters class.

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

def print_best_algorithm_code(individual):
    """Print usable code to run segmentation algorithm based on an
     individual's genetic representation vector."""
    #ind_algo = Segmentors.algoFromParams(individual)
    ind_algo = algoFromParams(individual)
    original_function = inspect.getsource(ind_algo.evaluate)

    # Get the body of the function
    function_contents = original_function[original_function.find('        '):\
                            original_function.find('return')]
    while function_contents.find('self.params') != -1:

        # Find the index of the 's' at the start of self.params
        params_index = function_contents.find('self.params')

        # Find the index of the ']' at the end of self.params["<SOME_TEXT>"]
        end_bracket_index = function_contents.find(']', params_index)+1

        # Find the first occurance of self.params["<SOME_TEXT>"] and store it
        code_to_replace = function_contents[params_index:end_bracket_index]

        # These offset will be used to access only the params_key
        offset = len('self.params["')
        offset2 = len('"]')

        # Get the params key
        params_key = function_contents[params_index + offset:end_bracket_index-offset2]

        # Use the params_key to access the params_value
        param_value = str(ind_algo.params[params_key])

        # Replace self.params["<SOME_TEXT>"] with the value of self.params["<SOME_TEXT>"]
        function_contents = function_contents.replace(code_to_replace, param_value)

    function_contents = function_contents.replace('        ', '')
    function_contents = function_contents[function_contents.find('\n\"\"\"')+5:]
    print(function_contents)
    return function_contents
   
def popCounts(pop):
    """Count the number of each algorihtm in a population"""
    algorithms = eval(parameters.ranges["algorithm"])
    counts = {a:0 for a in algorithms}
    for p in pop:
        #print(p[0])
        counts[p[0]] += 1
    return counts
        
        
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

    mxrange=256;
    #0
    ranges["algorithm"] = "['CT','FB','SC','WS','CV','MCV','AC']"
    descriptions["algorithm"] = "string code for the algorithm"
    
    #1
    descriptions["Channel"] = "A parameter for Picking the Channel R,G,B,H,S,V"
    ranges["Channel"] = "[0,1,2,3,4,5]"
    
    #2
    descriptions["MultiChannel"] = "True/False parameter"
    ranges["Channel"] = "[True, False]"   
    
    #3
    descriptions["alpha1"] = "General Purpos Lower bound threshold"
    ranges["alpha1"] = "[float(i)/256 for i in range(0,256)]"
    
    #4
    descriptions["alpha2"] = "General Purpos Upper bound threshold"
    ranges["alpha2"] = "[float(i)/256 for i in range(0,256)]"
    
    #5
    descriptions["beta1"] = "General Purpos Lower bound threshold"
    ranges["beta1"] = "[float(i)/256 for i in range(0,256)]"

    #6
    descriptions["beta2"] = "General Purpos Upper bound threshold"
    ranges["beta2"] = "[float(i)/256 for i in range(0,256)]"

    #7
    descriptions["gamma1"] = "General Purpos Lower bound threshold"
    ranges["gamma1"] = "[float(i)/256 for i in range(0,256)]"
    
    #8
    descriptions["gamma2"] = "General Purpos Upper bound threshold"
    ranges["gamma2"] = "[float(i)/256 for i in range(0,256)]"

    #9
    descriptions["n_segments"] = "General Purpos Upper bound threshold"
    ranges["n_segments"] = "[i for i in (range(0,10)]"

    #     Try to set defaults only once.
    #     Current method may cause all kinds of weird problems.
    #     @staticmethod
    #     def __Set_Defaults__()

    def __init__(self):
        """Set default values for each param in the dictionary."""
        self["algorithm"] = "None"
        self["Channel"] = 0
        self["alpha1"] = 0.5
        self["alpha2"] = 0.5
        self["beta1"] = 0.5
        self["beta2"] = 0.5
        self["gamma1"] = 0.5
        self["gamma2"] = 0.5
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
        self.altnames = []
        if paramlist:
            self.params.fromlist(paramlist)

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
    
    def evaluate(self, img):
        """Run segmentation algorithm to get inferred mask."""
        return np.zeros(img.shape[0:1])

    def __str__(self):
        """Return params for algorithm."""
        mystring = f"{self.params['algorithm']} -- \n"
        for p in self.paramindexes:
            mystring += f"\t{p} = {self.params[p]}\n"
        return mystring
        


class ColorThreshold(segmentor):
    """Peform Color Thresholding segmentation algorithm. Segments parts of the image
     based on the numerical values for the respective channel.

    Parameters:
    my_mx -- maximum thresholding value
    my_mn -- minimum thresholding value

    """

    def __init__(self, paramlist=None):
        """Get parameters from parameter list that are used in segmentation algorithm.
         Assign default values to these parameters."""
        super(ColorThreshold, self).__init__(paramlist)
        if not paramlist:
            self.params["algorithm"] = "CT"
            self.params["Channel"] = 5
            self.params["alpha1"] = 0.4
            self.params["alpha2"] = 0.6
        self.paramindexes = ["Channel", "alpha1", "alpha2"]
        self.altnames = ["Color Channel", "MinThreshold", "MaxThreshold"]
        self.checkparamindex()

    def evaluate(self, img): #XX
        """Evaluate segmentation algorithm on training image.

        Keyword arguments:
        img -- Original training image.

        Output:
        output -- resulting segmentation mask from algorithm.

        """
        channel_num = self.params["Channel"]
        if len(img.shape) > 2:
            num_channels = img.shape[2]
            if channel_num < num_channels:
                channel = img[:, :, int(channel_num)]
            else:
                hsv = skimage.color.rgb2hsv(img)
                #print(f"working with hsv channel {channel_num-3}")
                channel = hsv[:, :, int(channel_num)-3]
        else:
            channel = img
        pscale = np.max(channel)
        my_mx = self.params["alpha2"] * pscale
        my_mn = self.params["alpha1"] * pscale

        output = None
        
        if my_mn < my_mx:
            output = np.ones(channel.shape)
            output[channel < my_mn] = 0
            output[channel > my_mx] = 0
        else:
            output = np.zeros(channel.shape)
            output[channel > my_mn] = 1
            output[channel < my_mx] = 1

        return output


algorithmspace['CT'] = ColorThreshold

class TripleA (segmentor):
    def __init__(self, paramlist=None):
        super(TripleA, self).__init__(paramlist)
        if not paramlist:
            self.params["algorithm"] = "AAA"
            self.params["alpha1"] = 0.4
            self.params["alpha2"] = 0.6
        self.paramindexes = ["alpha1", "alpha2"]
        self.altnames = ["MinThreshold", "MaxThreshold"]
        self.checkparamindex()

    def evaluate(self, img): #XX
        channel_num = 1  # Do: Need to make this a searchable parameter.
        if len(img.shape) > 2:
            if channel_num < img.shape[2]:
                channel = img[:, :, 1]
            else:
                channel = img[:, :, 0]
        else:
            channel = img
        pscale = np.max(channel)
        my_mx = self.params["alpha2"] * pscale
        my_mn = self.params["alpha1"] * pscale
        if my_mx < my_mn:
            temp = my_mx
            my_mx = my_mn
            my_mn = temp

        output = np.ones(channel.shape)
        output[channel < my_mn] = 0
        output[channel > my_mx] = 0

        return output


algorithmspace["AAA"] = TripleA

class Felzenszwalb(segmentor):
    """Perform Felzenszwalb segmentation algorithm. ONLY WORKS FOR RGB. The felzenszwalb
     algorithms computes a graph based on the segmentation. Produces an oversegmentation
     of the multichannel using min-span tree. Returns an integer mask indicating the segment labels.

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
        """Get parameters from parameter list that are used in segmentation algorithm.
         Assign default values to these parameters."""
        super(Felzenszwalb, self).__init__(paramlist)
        if not paramlist:
            self.params["algorithm"] = "FB"
            self.params["alpha2"] = 0.984
            self.params["alpha1"] = 0.09
            self.params["beta1"] = 0.92
        self.paramindexes = ["alpha1", "alpha2", "beta1"]
        self.altnames = ["scale", "Stddev", "min_size"]
        self.checkparamindex()
        
    def evaluate(self, img):
        """Evaluate segmentation algorithm on training image.

        Keyword arguments:
        img -- Original training image.

        Output:
        output -- resulting segmentation mask from algorithm.

        """
        multichannel = self.params['MultiChannel']
        if multichannel:
            if len(img.shape) == 1:
                multichannel = False;
                channel=img;
        else:
            if len(img.shape) > 2:
                channel=img[:,:,self.params["channel"]]
            else:
                channel=img;

        
        if(multichannel):
            output = skimage.segmentation.felzenszwalb(
                img,
                self.params["alpha2"]*1000,
                self.params["alpha1"],
                self.params["beta1"]*100,
                multichannel=True,
            )
        else:
            output = skimage.segmentation.felzenszwalb(
                channel,
                self.params["alpha2"]*1000,
                self.params["alpha1"],
                self.params["beta1"]*100,
                multichannel=multichannel,
            )
        return output


algorithmspace["FB"] = Felzenszwalb

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

    """

    def __init__(self, paramlist=None):
        """Get parameters from parameter list that are used in segmentation algorithm.
         Assign default values to these parameters."""
        super(Slic, self).__init__(paramlist)
        if not paramlist:
            self.params["algorithm"] = "SC"
            self.params["n_segments"] = 5
            self.params["channel"] = 5
            #self.params["iterations"]
            self.params["alpha1"] = 0.5
        self.paramindexes = ["n_segments", "channel", "alpha1"]
        self.checkparamindex()
        
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
            compactness=10**(self.params["channel"]-3),
            max_iter=3,
            sigma=self.params["alpha1"],
            convert2lab=True,
            multichannel=multichannel,
        )
        return output


algorithmspace["SC"] = Slic

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

    """

    def __init__(self, paramlist=None):
        """Get parameters from parameter list that are used in segmentation algorithm.
         Assign default values to these parameters."""
        super(QuickShift, self).__init__(paramlist)
        if not paramlist:
            self.params["algorithm"] = "QS"
            self.params["kernel_size"] = 5
            self.params["max_dist"] = 60
            self.params["sigma"] = 5
            self.params["Channel"] = 1
            self.params["ratio"] = 2
        self.paramindexes = ["kernel_size", "max_dist", "sigma", "Channel", "ratio"]
        self.checkparamindex()
        
    def evaluate(self, img):
        """Evaluate segmentation algorithm on training image.

        Keyword arguments:
        img -- Original training image.

        Output:
        output -- resulting segmentation mask from algorithm.

        """
        output = skimage.segmentation.quickshift(
            color.gray2rgb(img),
            ratio=self.params["ratio"],
            kernel_size=self.params["kernel_size"],
            max_dist=self.params["max_dist"],
            sigma=self.params["sigma"],
            random_seed=self.params["Channel"],
        )
        return output


algorithmspace["QS"] = QuickShift

#DO: This algorithm seems to need a channel input. We should fix that.

class Watershed(segmentor):
    """Perform the Watershed segmentation algorithm. Uses user-markers.
     treats markers as basins and 'floods' them. Especially good if overlapping objects.
      Returns a labeled image ndarray.

    Parameters:
    image -- ndarray, input array
    compactness -- float, compactness of the basins. Higher values
        make more regularly-shaped basin.

    """

    # Not using connectivity, markers, or offset params as arrays would
    # expand the search space too much.
    # abbreviation for algorithm = WS

    def __init__(self, paramlist=None):
        """Get parameters from parameter list that are used in segmentation algorithm.
         Assign default values to these parameters."""
        super(Watershed, self).__init__(paramlist)
        if not paramlist:
            self.params["algorithm"] = "WS"
            self.params["compactness"] = 2.0
        self.paramindexes = ["compactness"]
        self.checkparamindex()
        
    def evaluate(self, img):
        """Evaluate segmentation algorithm on training image.

        Keyword arguments:
        img -- Original training image.

        Output:
        output -- resulting segmentation mask from algorithm.

        """
        channel = 0
        channel_img = img[:, :, channel]
        output = skimage.segmentation.watershed(
            channel_img, markers=None, compactness=self.params["compactness"]
        )
        return output


algorithmspace["WS"] = Watershed

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
        if not paramlist:
            self.params["algorithm"] = "CV"
            self.params["mu"] = 2.0
            self.params["lambda"] = (10, 20)
            self.params["iterations"] = 10
            self.params["dt"] = 0.10
            self.params["tolerance"] = 0.001
            self.params["init_level_set_chan"] = "small disk"
        self.paramindexes = ["mu", "lambda", "iterations", "dt", "init_level_set_chan"]
        self.checkparamindex()
        
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

    """

    # Abbreviation for algorithm = MCV

    def __init__(self, paramlist=None):
        """Get parameters from parameter list that are used in segmentation algorithm.
         Assign default values to these parameters."""
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
        self.checkparamindex()
        
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

    """

    # Abbrevieation for algorithm = AC

    def __init__(self, paramlist=None):
        """Get parameters from parameter list that are used in segmentation algorithm.
         Assign default values to these parameters."""
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
        self.checkparamindex()
        
    def evaluate(self, img):
        """Evaluate segmentation algorithm on training image.

        Keyword arguments:
        img -- Original training image.

        Output:
        output -- resulting segmentation mask from algorithm.

        """
        # We run the inverse_gaussian_gradient to get the image to use
        gimage = skimage.segmentation.inverse_gaussian_gradient(
            color.rgb2gray(img), self.params["alpha"], self.params["sigma"]
        )
        # zeros = 0
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

