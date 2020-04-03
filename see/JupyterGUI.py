"""This produces a GUI that allows users to switch between segmentation
 algorithms and alter the parameters manually using a slider. It shows two images,
  one with the original image with the resulting mask and one with the original image
   with the negative of the resulting mask."""
import matplotlib.pylab as plt
# from ipywidgets import interact
import ipywidgets as widgets

from see import Segmentors

def showtwo(img, img2):
    """Show two images side by side."""
    fig = plt.figure(figsize=(20, 20))
    my_ax = fig.add_subplot(1, 2, 1)
    my_ax.imshow(img)
    my_ax = fig.add_subplot(1, 2, 2)
    my_ax.imshow(img2)
    return fig

def showthree(img, img1, img2):
    """Show three images side by side."""
    fig = plt.figure(figsize=(20, 20))
    my_ax = fig.add_subplot(1, 3, 1)
    my_ax.imshow(img)
    my_ax = fig.add_subplot(1, 3, 2)
    my_ax.imshow(img1)
    my_ax = fig.add_subplot(1, 3, 3)
    my_ax.imshow(img2)
    return fig

def show_segment(img, mask):
    """Show both options for segmenting using the current mask.

    Keyword arguments:
    img -- original image
    mask -- resulting mask from segmentor

    """
    im1 = img.copy()
    im2 = img.copy()
    im1[mask > 0, :] = 0
    im2[mask == 0, :] = 0
    fig = showtwo(im1, im2)
    return fig


def segmentwidget(img, gmask, params=None, alg=None):
    """Generate GUI. Produce slider for each parameter for the current segmentor.
     Show both options for the masked image.

    Keyword arguments:
    img -- original image
    gmask -- ground truth segmentation mask for the image
    params -- list of parameter options
    alg -- algorithm to search parameters over

    """
    if params==None and alg == None:
        alg = 'FB'
        params = [alg, 0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, (1, 1), 0, 'checkerboard', 'checkerboard', 0, 0, 0, 0, 0, 0]
    elif params == None and alg != None:
        params = [alg, 0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, (1, 1), 0, 'checkerboard', 'checkerboard', 0, 0, 0, 0, 0, 0]
    elif params != None and alg != None:
        params[0] = alg
    seg = Segmentors.algoFromParams(params)
    widg = dict()
    widglist = []

    for p in seg.paramindexes:
        thislist = eval(seg.params.ranges[p])
        # disabled = True
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

    def func(img=img, mask=gmask, **kwargs):
        """Find mask and fitness for current algorithm. Show masked image."""
        print(seg.params["algorithm"])
        for k in kwargs:
            seg.params[k] = kwargs[k]
        mask = seg.evaluate(img)
        fit = Segmentors.FitnessFunction(mask, gmask)
        # fig = show_segment(img, mask)
        fig = showtwo(img, mask)
        plt.title('Fitness Value: ' + str(fit[0]))


    layout = widgets.Layout(grid_template_columns='1fr 1fr 1fr')
    u_i = widgets.GridBox(widglist, layout=layout)

    out = widgets.interactive_output(func, widg)
    display(u_i, out)
