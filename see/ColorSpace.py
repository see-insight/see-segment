"""ColorSpace.py file."""

from skimage import color
from base_classes import param_space, algorithm
import copy
import inspect
import random

import sys
import logging
import numpy as np
import skimage

# TODO My guess is this algorithm is very slow.  We need to use a cache to speed it up.


class color_params(param_space):
    """Class color_params."""
    
    descriptions = dict()
    ranges = dict()
    pkeys = []


color_params.add('colorspace',
                 ['RGB', 'HSV', 'RGB CIE', 'XYZ', 'YUV',
                     'YIQ', 'YPbPr', 'YCbCr', 'YDbDr'],
                 "Pick a colorspace [‘RGB’, ‘HSV’, ‘RGB CIE’, ‘XYZ’, ‘YUV’, ‘YIQ’, ‘YPbPr’, ‘YCbCr’, ‘YDbDr’]"
                 )
color_params.add('multichannel',
                 [True, False],
                 "True/False parameter"
                 )
color_params.add('channel',
                 [0, 1, 2],
                 "A parameter for Picking the Channel 0,1,2"
                 )


class colorspace(algorithm):
       """Class colorspace."""
        
    def getchannel(img, colorspace, channel):
        """Function that returns a single channel from an image.
        ['RGB', ‘HSV’, ‘RGB CIE’, ‘XYZ’, ‘YUV’, ‘YIQ’, ‘YPbPr’, ‘YCbCr’, ‘YDbDr’]
        """
        dimention = 3
        if (len(img.shape) == 2):
            c_img = img.copy()
            img = np.zeros([c_img.shape[0], c_img.shape[1], 3])
            img[:, :, 0] = c_img
            img[:, :, 1] = c_img
            img[:, :, 2] = c_img
            return [img, c_img, 1]

        if(colorspace == 'RGB'):
            return [img, img[:, :, channel], 3]
        else:
            space = color.convert_colorspace(img, 'RGB', colorspace)
            return [space, space[:, :, channel], 3]

    # TODO Update to allow paramlist to be either a list or the parameters class
    def __init__(self, paramlist=None):
        """Generate algorithm params from parameter list."""
        # init_params()
        self.params = color_params()
        self.params['colorspace'] = 'RGB'
        self.params['multichannel'] = True
        self.params['channel'] = 2

        self.chache = dict()
        if paramlist:
            if (type(paramlist) == list):
                self.params.fromlist(paramlist)
            else:
                self.params = paramlist
        else:
            self.params["multichannel"] = True
            self.params["colorspace"] = "RGB"
            self.params["channel"] = 2
        self.paramindexes = ["colorspace", "multichannel", "channel"]
        self.checkparamindex()

    # TODO use name to build a dictionary to use as a chache
    def evaluate(self, img, name=None):
        """Run segmentation algorithm to get inferred mask."""
        multichannel = self.params['multichannel']

        if len(img.shape) > 2:
            multichannel = False

        [img, channel, dimention] = colorspace.getchannel(
            img, self.params['colorspace'], self.params['channel'])

        if multichannel:
            return img
        else:
            return channel

    def pipe(self, data):
        """Set inputimage and img to evaluated data images."""
        data.inputimage = data.img
        data.img = self.evaluate(data.img)
        return data
