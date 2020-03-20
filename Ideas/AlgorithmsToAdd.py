""" Extra algorithms or functions that are not ready to
be included, or we don't want to include yet.
"""


class Flood(segmentor):
    '''
    #flood
    #DOES NOT SUPPORT MULTICHANNEL IMAGES
    https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_floodfill.html
    Uses a seed point and to fill all connected points within/equal to
        a tolerance around the seed point
    #Returns a boolean array with 'flooded' areas being true
    #Variables
    image: ndarray, input image
    seed_point: tuple/int, x,y,z referring to starting point for flood
        fill
    selem: ndarray of 1's and 0's, Used to determine neighborhood of
        each pixel
    connectivity: int, Used to find neighborhood of each pixel. Can use
        this or selem.
    tolerance: float or int, If none, adjacent values must be equal to
        seed_point. Otherwise, how likely adjacent values are flooded.
    '''
    #Abbreviation for algorithm = FD

    def __init__(self, paramlist=None):
        super(Flood, self).__init__(paramlist)
        self.params['algorithm'] = 'AC'
        self.params['seed_pointX'] = 10
        self.params['seed_pointY'] = 20
        self.params['seed_pointZ'] = 0
        self.params['connect'] = 4
        self.params['tolerance'] = 0.5
        self.paramindexes = ['seed', 'connect', 'tolerance']

    def evaluate(self, img):
        output = skimage.segmentation.flood(
            img,
            (self.params['seed_pointX'],
             self.params['seed_pointY'],
             self.params['seed_pointZ']),
            connectivity=self.params['connect'],
            tolerance=self.params['tolerance'])
        return output
algorithmspace['FD'] = Flood


class FloodFill(segmentor):
    '''
    #flood_fill
    https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_floodfill.html
    Like a paint-bucket tool in paint. Like flood, but changes the
        color equal to new_type
    #Returns A filled array of same shape as the image
    #Variables
    image: ndarray, input image
    seed_point: tuple or int, starting point for filling (x,y,z)
    new_value: new value to set the fill to (e.g. color). Must agree
        with image type
    selem: ndarray, Used to find neighborhood of filling
    connectivity: Also used to find neighborhood of filling if selem is
        None
    tolerance: float or int, If none, adjacent values must be equal to
        seed_point. Otherwise, how likely adjacent values are flooded.
    inplace: bool, If true, the flood filling is applied to the image,
        if False, the image is not modified. Default False, don't
        change
    '''
    #Abbreviation for algorithm == FF

    def __init__(self, paramlist=None):
        super(FloodFill, self).__init__(paramlist)
        self.params['algorithm'] = 'AC'
        self.params['seed_pointX'] = 10
        self.params['seed_pointY'] = 20
        self.params['seed_pointZ'] = 0
        self.params['connect'] = 4
        self.params['tolerance'] = 0.5
        self.paramindexes = ['seed', 'connect', 'tolerance']

    def evaluate(self, img):
        output = skimage.segmentation.flood_fill(
            img,
            (self.params['seed_pointX'],
             self.params['seed_pointY'],
             self.params['seed_pointZ']),
            134,  #TODO: Had coded value
            connectivity= self.params['connect'],
            tolerance=self.params['tolerance'])
        try:
            #I'm not sure if this will work on grayscale
            image = Image.fromarray(output.astype('uint8'), '1')
        except ValueError:
            image = Image.fromarray(output.astype('uint8'), 'RGB')

        width = image.width
        height = image.width


        #Converting the background to black
        for x in range(0, width):
            for y in range(0, height):
                #First check for grayscale
                pixel = image.getpixel((x,y))
                if pixel[0] == 134:
                    image.putpixel((x,y), 134)
                    continue
                else:
                    image.putpixel((x,y), 0)
                    #print(image.getpixel((x,y)))

        #image.convert(mode='L')
        pic = np.array(image)
        return pic
algorithmspace['FF'] = FloodFill

# TODO: Figure out the mask part?
class RandomWalker(segmentor):
    algorithm = 'RW'
    paramindexes = [1, 2]

    def __doc__(self):
        myhelp = "Wrapper function for the scikit-image random_walker segmentor:"
        myhelp += f" xx {skimage.segmentation.random_walker.__doc__}"
        return myhelp

    def __init__(self, beta = 0.5, tolerance = 0.4):
        self.beta = beta
        self.tolerance = tolerance

    def evaluate(self, img):
        #Let's deterime what mode to use
        mode = "bf"
        if len(img) < 512 :
            mode = "cg_mg"

        #If data is 2D, then this is a grayscale, so multichannel is
        output = skimage.segmentation.random_walker(
            img, labels=mask,
            beta=self.beta,
            tol=self.tolerance, copy=True,
            multichannel=True, return_full_prob=False)
        return output