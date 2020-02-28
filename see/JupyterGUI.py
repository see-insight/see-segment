"""This produces a GUI that allows users to switch between segmentation algorithms and alter the parameters manually using a slider. It shows two images, one with the original image with the resulting mask and one with the original image with the negative of the resulting mask."""
import matplotlib.pylab as plt
from ipywidgets import interact
import ipywidgets as widgets

from see import Segmentors

def showtwo(img, img2):
    """Show two images side by side."""
    fig = plt.figure(figsize=(20,20))
    ax = fig.add_subplot(1,2,1)
    ax.imshow(img)
    ax = fig.add_subplot(1,2,2)
    ax.imshow(img2)
    return fig
    
def showthree(im, img, img2):
    """Show three images side by side."""
    fig = plt.figure(figsize=(20,20))
    ax = fig.add_subplot(1,3,1)
    ax.imshow(im)
    ax = fig.add_subplot(1,3,2)
    ax.imshow(img)
    ax = fig.add_subplot(1,3,3)
    ax.imshow(img2)
    return fig
    
def showSegment(im, mask):
    """Show both options for segmenting using the current mask.

    Keyword arguments:
    im -- original image
    mask -- resulting mask from segmentor

    """
    im1 = im.copy()
    im2 = im.copy()
    im1[mask>0,:] = 0
    im2[mask==0,:] = 0
    fig = showtwo(im1,im2)
    return fig


def segmentwidget(params, img, gmask):
    """Generate GUI. Produce slider for each parameter for the current segmentor. Show both options for the masked image.

    Keyword arguments:
    params -- list of parameter options
    img -- original image
    gmask -- ground truth segmentation mask for the image

    """
    seg = Segmentors.algoFromParams(params)
    widg = dict()
    widglist = []

    for p in seg.paramindexes:
        thislist = eval(seg.params.ranges[p])
        disabled = True
        thiswidg = widgets.SelectionSlider(options=tuple(thislist),
                                           disabled=False,
                                           description=p,
                                           value=seg.params[p],
                                           continuous_update=False,
                                           orientation='horizontal',
                                           readout=True
                                          )
        widglist.append(thiswidg)
        widg[p] = thiswidg

    def f(im =img, mask=gmask, **kwargs):
        """Find mask and fitness for current algorithm. Show masked image."""
        print(seg.params["algorithm"])
        for k in kwargs:
            seg.params[k] = kwargs[k]
        mask = seg.evaluate(img)
        fit = Segmentors.FitnessFunction(mask,gmask)
        fig = showSegment(img,mask)
        plt.title(fit)
        

    layout = widgets.Layout(grid_template_columns='1fr 1fr 1fr')
    ui = widgets.GridBox(widglist, layout=layout)

    out = widgets.interactive_output(f, widg)
    display(ui, out)

    
    