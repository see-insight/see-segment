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
    ranges["algorithm"] = "['ColorThreshold','Felzenszwalb','Slic', 'SlicO', 'QuickShift', 'Watershed', 'Chan_Vese','Morphological_Chan_Vese,'MorphGeodesicActiveContour']"
    
    descriptions["algorithm"] = "string code for the algorithm"

    #1
    descriptions["multichannel"] = "True/False parameter"
    ranges["multichannel"] = "[True, False]"   
    
    #2
    #TODO: Change this to the actual strings to make the param space easier to read
    descriptions["colorspace"] = "Pick a colorspace [‘RGB’, ‘HSV’, ‘RGB CIE’, ‘XYZ’, ‘YUV’, ‘YIQ’, ‘YPbPr’, ‘YCbCr’, ‘YDbDr’]"
    ranges["colorspace"] = "['RGB', 'HSV', 'RGB CIE', 'XYZ', 'YUV', 'YIQ', 'YPbPr', 'YCbCr', 'YDbDr']"
    
    #3
    descriptions["channel"] = "A parameter for Picking the Channel 0,1,2"
    ranges["channel"] = "[0,1,2]"
    
    #4
    descriptions["alpha1"] = "General Purpos Lower bound threshold"
    ranges["alpha1"] = "[float(i)/256 for i in range(0,256)]"
    
    #5
    descriptions["alpha2"] = "General Purpos Upper bound threshold"
    ranges["alpha2"] = "[float(i)/256 for i in range(0,256)]"
    
    #6
    descriptions["beta1"] = "General Purpos Lower bound threshold"
    ranges["beta1"] = "[float(i)/256 for i in range(0,256)]"

    #7
    descriptions["beta2"] = "General Purpos Upper bound threshold"
    ranges["beta2"] = "[float(i)/256 for i in range(0,256)]"

    #8
    descriptions["gamma1"] = "General Purpos Lower bound threshold"
    ranges["gamma1"] = "[float(i)/256 for i in range(0,256)]"
    
    #9
    descriptions["gamma2"] = "General Purpos Upper bound threshold"
    ranges["gamma2"] = "[float(i)/256 for i in range(0,256)]"

    #10
    descriptions["n_segments"] = "General Purpos Upper bound threshold"
    ranges["n_segments"] = "[i for i in range(0,10)]"
    
    #11
    descriptions["max_iter"] = "General Purpos Upper bound threshold"
    ranges["max_iter"] = "[i for i in range(1,20)]"
    
    
    #     Try to set defaults only once.
    #     Current method may cause all kinds of weird problems.
    #     @staticmethod
    #     def __Set_Defaults__()

    def __init__(self):
        """Set default values for each param in the dictionary."""
        self["algorithm"] = "None"
        self["multichannel"] = False
        self["colorspace"] = "HSV"
        self["channel"] = 2
        self["alpha1"] = 0.5
        self["alpha2"] = 0.5
        self["beta1"] = 0.5
        self["beta2"] = 0.5
        self["gamma1"] = 0.5
        self["gamma2"] = 0.5
        self["n_segments"] = 3
        self["max_iter"] = 10
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

    def __init__(self, paramlist=None):
        """Get parameters from parameter list that are used in segmentation algorithm.
         Assign default values to these parameters."""
        super(ColorThreshold, self).__init__(paramlist)
        if not paramlist:
            self.params["algorithm"] = "ColorThreshold"
            self.params["multichannel"] = False
            self.params["colorspace"] = "HSV"
            self.params["channel"] = 2
            self.params["alpha1"] = 0.4
            self.params["alpha2"] = 0.6
            self.params["beta1"] = 0.4
            self.params["beta2"] = 0.6
            self.params["gamma1"] = 0.4
            self.params["gamma2"] = 0.6
        self.paramindexes = ["multichannel", "colorspace", "channel", 
                             "alpha1", "alpha2", 
                             "beta1", "beta2", 
                             "gamma1", "gamma2"]
        self.checkparamindex()

    def evaluate(self, input_img): #XX
        """Evaluate segmentation algorithm on training image.

        Keyword arguments:
        img -- Original training image.

        Output:
        output -- resulting segmentation mask from algorithm.

        """
        [img, channel, dimention] = getchannel(
            input_img, self.params["colorspace"], self.params["channel"]
        ) 

        minlist = ["alpha1", "beta1", "gamma1"]
        maxlist = ["alpha2", "beta2", "gamma2"]
        
        output = None
        
        if (self.params["multichannel"] and dimention > 1):
            output = np.ones([img.shape[0], img.shape[1]])
            for dimidx in range(3):
                pscale = np.max(img[:,:,dimidx])
                my_mn = self.params[minlist[dimidx]] * pscale  
                my_mx = self.params[maxlist[dimidx]] * pscale
                               
                if my_mn < my_mx:
                    output[img[:,:,dimidx] < my_mn] = 0
                    output[img[:,:,dimidx] > my_mx] = 0
                else:
                    flag1 = img[:,:,dimidx] > my_mn
                    flag2 = img[:,:,dimidx] < my_mx
                    output[np.logical_and(flag1,flag2)] = 0
        else:
            pscale = np.max(channel)
            chidx = self.params["channel"]
            my_mx = self.params[maxlist[chidx]] * pscale
            my_mn = self.params[minlist[chidx]] * pscale              

            if my_mn < my_mx:
                output = np.ones(channel.shape)
                output[channel < my_mn] = 0
                output[channel > my_mx] = 0
            else:
                output = np.zeros(channel.shape)
                output[channel > my_mn] = 1
                output[channel < my_mx] = 1
        return output

algorithmspace['ColorThreshold'] = ColorThreshold

# class TripleA (segmentor):
#     def __init__(self, paramlist=None):
#         super(TripleA, self).__init__(paramlist)
#         if not paramlist:
#             self.params["algorithm"] = "AAA"
#             self.params["alpha1"] = 0.4
#             self.params["alpha2"] = 0.6
#         self.paramindexes = ["alpha1", "alpha2"]
#         #self.altnames = ["MinThreshold", "MaxThreshold"]
#         self.checkparamindex()

#     def evaluate(self, img): #XX
#         channel = getchannel(img, self.params["channel"])
#         pscale = np.max(channel)
#         my_mx = self.params["alpha2"] * pscale
#         my_mn = self.params["alpha1"] * pscale
#         if my_mx < my_mn:
#             temp = my_mx
#             my_mx = my_mn
#             my_mn = temp

#         output = np.ones(channel.shape)
#         output[channel < my_mn] = 0
#         output[channel > my_mx] = 0

#         return output


# algorithmspace["AAA"] = TripleA

class Felzenszwalb(segmentor):
    """Perform Felzenszwalb segmentation algorithm. The felzenszwalb algorithms computes a 
    graph based on the segmentation. Produces an oversegmentation of the multichannel using 
    min-span tree. Returns an integer mask indicating the segment labels.
    
    Note: a colorspace of 'HSV' and a channel of 2 is a grayscale image. 
    
    https://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.felzenszwalb

    Parameters:
    mulitchannel - (multichannel) - bool, Whether the image is 2D or 3D
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
        if not paramlist:
            self.params["multichannel"]=True
            self.params["colorspace"] = 'RGB'
            self.params["channel"]=2
            self.params["algorithm"] = "Felzenszwalb"
            self.params["alpha2"] = 0.984
            self.params["alpha1"] = 0.09
            self.params["beta1"] = 0.92
        self.paramindexes = ["multichannel", "colorspace", "channel", "alpha1", "alpha2", "beta1"]
        self.checkparamindex()
        
    def evaluate(self, input_img):
        """Evaluate segmentation algorithm on training image.

        Keyword arguments:
        img -- Original training image.

        Output:
        output -- resulting segmentation mask from algorithm.

        """

            
        scale = self.params["alpha2"]*1000
        sigma = self.params["alpha1"]
        min_size = int(self.params["beta1"]*100)
        
        [img, channel, dimention] = getchannel(
            input_img, self.params["colorspace"], self.params["channel"]
        ) 
        
        if(self.params["multichannel"] and dimention > 1):
            output = skimage.segmentation.felzenszwalb(
                img,
                scale,
                sigma,
                min_size,
                multichannel=True
            )
        else:
            output = skimage.segmentation.felzenszwalb(
                channel,
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
        
        
        
        
algorithmspace["Felzenszwalb"] = Felzenszwalb

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
        if not paramlist:
            self.params["algorithm"] = "Slic"
            self.params["multichannel"]=True
            self.params["colorspace"] = 'HSV'
            self.params["n_segments"] = 5
            self.params["channel"] = 2
            self.params["max_iter"] = 10
            self.params["alpha1"] = 0.5
        self.paramindexes = ["multichannel", "colorspace", "channel", "n_segments", "alpha1", "max_iter"]
        self.checkparamindex()
        self.slico = False
        
    def evaluate(self, input_img):
        """Evaluate segmentation algorithm on training image.

        Keyword arguments:
        img -- Original training image.

        Output:
        output -- resulting segmentation mask from algorithm.

        """

        [img, channel, dimention] = getchannel(
            input_img, self.params["colorspace"], self.params["channel"]
        )                     

        compactness=10**(self.params["channel"]-3)
        n_segments = self.params["n_segments"]+1
        max_iter=self.params["max_iter"]
        if(self.params["multichannel"] and dimention > 1):            
            output = skimage.segmentation.slic(
                img,
                n_segments=n_segments,
                compactness=compactness,
                max_iter=max_iter,
                sigma=0, # Gaussian smoothing should happen as a preprocessing step.
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


algorithmspace["Slic"] = Slic

#TODO Update to remove any parameters that SLICO dosn't use. (Currently this includes the SLIP parameters)

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

algorithmspace["SlicO"] = SlicO

#TODO Quickshift is very slow, we need to do some benchmarks and see what are resonable running ranges.

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
        if not paramlist:
            self.params["algorithm"] = "QuickShift"
            self.params["colorspace"]= "HSV"
            self.params["channel"] = 2
            self.params["alpha1"] = 0.5
            self.params["beta1"] = 0.5
            self.params["beta2"] = 0.5

        self.paramindexes = ["colorspace", "channel", "alpha1", "beta1", "beta2" ]
        self.checkparamindex()
        
    def evaluate(self, input_img):
        """Evaluate segmentation algorithm on training image.

        Keyword arguments:
        img -- Original training image.

        Output:
        output -- resulting segmentation mask from algorithm.

        """
        
        [img, channel, dimention] = getchannel(
            input_img, self.params["colorspace"], self.params["channel"]
        ) 
        
        mindim = min(channel.shape)
        
        ratio = self.params["alpha1"]
        kernel_size = mindim/10*self.params["beta1"]
        max_dist = mindim*self.params["beta2"]
        output = skimage.segmentation.quickshift(
            img,
            ratio=ratio,
            kernel_size=kernel_size,
            max_dist=max_dist,
            sigma=0, # TODO this should be handeled in the preprocessing step
            random_seed=1,
        )
        return output


algorithmspace["QuickShift"] = QuickShift

#TODO Watershed one seems to be broken all we get is a line at the top.

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
        if not paramlist:
            self.params["algorithm"] = "Watershed"
            self.params["colorspace"]= "HSV"
            self.params["channel"] = 2
            self.params["alpha1"] = 0.66
        self.paramindexes = ["colorspace", "channel", "alpha1"]
        self.checkparamindex()
        
    def evaluate(self, input_img):
        """Evaluate segmentation algorithm on training image.

        Keyword arguments:
        img -- Original training image.

        Output:
        output -- resulting segmentation mask from algorithm.
        
        

        """
        [img, channel, dimention] = getchannel(
            input_img, self.params["colorspace"], self.params["channel"]
        ) 
        
        compactness=self.params["alpha1"]*3
        
        output = skimage.segmentation.watershed(
            img, markers=None, compactness=compactness
        )
        return output


algorithmspace["Watershed"] = Watershed

#TODO Chan_Vese one seems very broken.  All we get is a circle.

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
            self.params["algorithm"] = "Chan_Vese"
            self.params["colorspace"]= "HSV"
            self.params["channel"] = 2
            self.params["alpha1"] = 1
            self.params["beta1"] = 1
            self.params["beta2"] = 1
            self.params["max_iter"] = 10
            self.params["alpha2"] = 0.10
            self.params["n_segments"] = 0
            #self.params["tolerance"] = 0.001 #TODO Removed, consider adding in later if need be.
        self.paramindexes = ["colorspace", "channel", "alpha1", "alpha2", "beta1", "beta2", "n_segments", "max_iter"]
        self.checkparamindex()
        
    def evaluate(self, input_img):
        """Evaluate segmentation algorithm on training image.

        Keyword arguments:
        img -- Original training image.

        Output:
        output -- resulting segmentation mask from algorithm.

        """
        
        [img, channel, dimention] = getchannel(
            input_img, self.params["colorspace"], self.params["channel"]
        ) 
        
        mu= self.params["alpha1"]*2 #TODO I think this should be between zero and one.
        lambda1 = self.params["beta1"] #TODO Not sure about the range of these. Previous was (10,20)
        lambda2 = self.params["beta2"]
        max_iter = self.params["max_iter"]
        dt = self.params["alpha2"]
        
        level_set_shapes= ['checkerboard', 'disk', 'small disk']
        init_level_set = level_set_shapes[self.params['n_segments']%3]
        
        output = skimage.segmentation.chan_vese(
            channel,
            mu=mu,
            lambda1=lambda1,
            lambda2=lambda2,
            max_iter=max_iter,
            dt=dt,
            init_level_set=init_level_set
            
        )
        return output


algorithmspace["Chan_Vese"] = Chan_Vese

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
            self.params["colorspace"]= "HSV"
            self.params["channel"] = 2
            self.params["alpha1"] = 1
            self.params["beta1"] = 1
            self.params["beta2"] = 1
            self.params["max_iter"] = 10
            self.params["n_segments"] = 0
            #self.params["tolerance"] = 0.001 #TODO Removed, consider adding in later if need be.
        self.paramindexes = ["colorspace", "channel", "alpha1",  "beta1", "beta2", "n_segments", "max_iter"]
        self.checkparamindex()
        
    def evaluate(self, input_img):
        """Evaluate segmentation algorithm on training image.

        Keyword arguments:
        img -- Original training image.

        Output:
        output -- resulting segmentation mask from algorithm.

        """
                
        [img, channel, dimention] = getchannel(
            input_img, self.params["colorspace"], self.params["channel"]
        ) 
        
        smoothing= int(self.params["alpha1"]*4) #TODO We may want to move this? We need a number 1-4 smoothing iterations
        
        lambda1 = self.params["beta1"] #TODO Not sure about the range of these. Previous was (10,20)
        lambda2 = self.params["beta2"]
        max_iter = self.params["max_iter"]
        level_set_shapes= ['checkerboard', 'circle']
        init_level_set = level_set_shapes[self.params['n_segments']%2]
        
        
        
        
        if len(img.shape) == 3:
            img = skimage.color.rgb2gray(img)
        output = skimage.segmentation.morphological_chan_vese(
            img,
            iterations=max_iter,
            init_level_set=init_level_set,
            smoothing=smoothing,
            lambda1=lambda1,
            lambda2=lambda2,
        )
        return output


algorithmspace["Morphological_Chan_Vese"] = Morphological_Chan_Vese

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
            self.params["colorspace"]= "HSV"
            self.params["channel"] = 2
            self.params["alpha1"] = 1
            self.params["alpha2"] = 1
            self.params["beta1"] = 0.2
            self.params["beta2"] = 0.3
            self.params["beta2"] = 1
            self.params["max_iter"] = 10
            self.params["n_segments"] = 0
            #self.params["tolerance"] = 0.001 #TODO Removed, consider adding in later if need be.
        self.paramindexes = ["colorspace", "channel", "alpha1",  "alpha2", "beta1", "beta2", "n_segments", "max_iter"]
        self.checkparamindex()
        
    def evaluate(self, input_img):
        """Evaluate segmentation algorithm on training image.

        Keyword arguments:
        img -- Original training image.

        Output:
        output -- resulting segmentation mask from algorithm.

        """
        
        [img, channel, dimention] = getchannel(
            input_img, self.params["colorspace"], self.params["channel"]
        ) 
        
        smoothing= int(self.params["alpha1"]*4) #TODO We may want to move this? We need a number 1-4 smoothing iterations
        balloon = (self.params["alpha2"]*100)-50
        max_iter = self.params["max_iter"]
        level_set_shapes= ['checkerboard', 'circle']
        init_level_set = level_set_shapes[self.params['n_segments']%2]

        
        # We run the inverse_gaussian_gradient to get the image to use
        gimage = skimage.segmentation.inverse_gaussian_gradient(
            color.rgb2gray(img), self.params["beta1"], self.params["beta2"]
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

algorithmspace["MorphGeodesicActiveContour"] = MorphGeodesicActiveContour

