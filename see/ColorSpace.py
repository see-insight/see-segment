"""ColorSpace.py file."""

from skimage import color
from see.base_classes import param_space, algorithm, pipedata
import copy
import inspect
import random

import sys
import logging
import numpy as np
import skimage


# TODO My guess is this algorithm is very slow.  We need to use a cache to
# speed it up.


class color_params(param_space):
    """Class color_params."""
    
    descriptions = dict()
    ranges = dict()
    pkeys = []


color_params.add(
    'colorspace',
    [
        'RGB',
        'HSV',
        'RGB CIE',
        'XYZ',
        'YUV',
        'YIQ',
        'YPbPr',
        'YCbCr',
        'YDbDr'],
    "Pick a colorspace [‘RGB’, ‘HSV’, ‘RGB CIE’, ‘XYZ’, ‘YUV’, ‘YIQ’, ‘YPbPr’, ‘YCbCr’, ‘YDbDr’]")
color_params.add('multichannel',
                 [True, False],
                 "True/False parameter"
                 )
color_params.add('channel',
                 [0, 1, 2],
                 "A parameter for Picking the Channel 0,1,2"
                 )

def selectcolorspace(img, multichannel=True, colorspace='RGB', channel=2):
    """Function that takes an image as an input converts it to a new colorspace and
       returns a channel from that colorspace. Colorspaces include the following: 
    ['RGB', ‘HSV’, ‘RGB CIE’, ‘XYZ’, ‘YUV’, ‘YIQ’, ‘YPbPr’, ‘YCbCr’, ‘YDbDr’]
    """
    import numpy as np
    from skimage import color
    if len(img.shape) == 2:
        channelimg = img.copy()
        space = np.zeros([channelimg.shape[0], channelimg.shape[1], 3])
        space[:, :, 0] = channelimg 
        space[:, :, 1] = channelimg
        space[:, :, 2] = channelimg
    else: 
        # If image is grater than 3 channels cut it down to three
        if img.shape[2] > 3:
            ###TODO: Fix this hack to make a image with more than 3 channels a 3 channel image
            img = img[:,:,0:3]
        if colorspace == 'RGB':
            ### Assume the input 3channel colorspace is RGB and dosn't need changing
            space = img
        else:
            ### use the color library convert_colorspace fucntion 
            space = color.convert_colorspace(img, 'RGB', colorspace)

        channelimg = img[:, :, channel]
    return space if multichannel else channelimg

class colorspace(algorithm):
    """colorspace."""
        


    # TODO Update to allow paramlist to be either a list or the parameters
    # class
    def __init__(self, paramlist=None):
        """Generate algorithm params from parameter list."""
        # init_params()
        self.params = color_params()
        self.params['colorspace'] = 'RGB'
        self.params['multichannel'] = True
        self.params['channel'] = 2

        self.chache = dict()
        if paramlist:
            if isinstance(paramlist, list):
                self.params.fromlist(paramlist)
            else:
                self.params = paramlist
        else:
            self.params["multichannel"] = True
            self.params["colorspace"] = "RGB"
            self.params["channel"] = 2
        self.paramindexes = ["colorspace", "multichannel", "channel"]
        self.checkparamindex()
    
    def evaluate(self, img, name=None):
        """Run segmentation algorithm to get inferred mask."""
        multichannel = self.params['multichannel']
        colorspace = self.params['colorspace']
        channel = self.params['channel']
        return selectcolorspace(img, multichannel=multichannel, colorspace=colorspace, channel=channel)

    def pipe(self, data):
        """Set inputimage and img to evaluated data images."""
        if type(data) is pipedata:
            data.append(self.evaluate(data[-1]))
        else:
            for dataimage in data:
                this.pipe(data)
        return data
    
    def algorithm_code(self):
        """Print usable code to run segmentation algorithm.
        
        Based on an
        individual's genetic representation vector.
        """
        original_function = inspect.getsource(self.getchannel)
        
        original_function += inspect.getsource(self.evaluate)

        return original_function
