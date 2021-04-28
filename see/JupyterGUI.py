"""This produces a GUI that allows users to switch between segmentation
 algorithms and alter the parameters manually using a slider. It shows two images,
  one with the original image with the resulting mask and one with the original image
   with the negative of the resulting mask."""
import matplotlib.pylab as plt
import ipywidgets as widgets
from IPython.display import display, clear_output

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

def picksegment(algorithms):
    w = widgets.Dropdown(
        options=algorithms,
        value=algorithms[0],
        description='Choose Algorithm:',
    )

    def on_change(change):
        if change['type'] == 'change' and change['name'] == 'value':
            clear_output(wait=True) # Clear output for dynamic display
            display(w)
            print(help(Segmentors.algorithmspace[change['new']]))
    w.observe(on_change)

    display(w)
    print(help(Segmentors.algorithmspace[w.value]))
    return w

def segmentwidget(img, gmask, params=None, alg=None):
    """Generate GUI. Produce slider for each parameter for the current segmentor.
     Show both options for the masked image.

    Keyword arguments:
    img -- original image
    gmask -- ground truth segmentation mask for the image
    params -- list of parameter options
    alg -- algorithm to search parameters over

    """
    if params is None and alg is None:
        alg = 'FB'
        params = [alg, 0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,\
         (1, 1), 0, 'checkerboard', 'checkerboard', 0, 0, 0, 0, 0, 0]
    elif params is None and alg is not None:
        params = [alg, 0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,\
         (1, 1), 0, 'checkerboard', 'checkerboard', 0, 0, 0, 0, 0, 0]
    elif params is not None and alg is not None:
        params[0] = alg
    seg = Segmentors.algoFromParams(params)
    widg = dict()
    widglist = []

    for ppp, ind in zip(seg.paramindexes, range(len(seg.paramindexes))):
        thislist = eval(seg.params.ranges[ppp])
        if (ind < len(seg.altnames)):
            name = seg.altnames[ind]
        else:
            name = ppp
           
        thiswidg = widgets.SelectionSlider(options=tuple(thislist),
                                           disabled=False,
                                           description=name,
                                           value=seg.params[ppp],
                                           continuous_update=False,
                                           orientation='horizontal',
                                           readout=True
                                          )
        widglist.append(thiswidg)
        widg[ppp] = thiswidg

#     algorithms = list(Segmentors.algorithmspace.keys())
#     w = widgets.Dropdown(
#         options=algorithms,
#         value=algorithms[0],
#         description='Choose Algorithm:',
#     )
    

    
    def func(img=img, mask=gmask, **kwargs):
        """Find mask and fitness for current algorithm. Show masked image."""
        print(seg.params["algorithm"])
        for k in kwargs:
            seg.params[k] = kwargs[k]
        mask = seg.evaluate(img)
        fit = Segmentors.FitnessFunction(mask, gmask)
        fig = showtwo(img, mask)
#         plt.title('Fitness Value: ' + str(fit[0]))
        print(help(seg))

#     def on_change(change):
#         display(w, u_i, out)
#     w.observe(on_change)
    
    layout = widgets.Layout(grid_template_columns='1fr 1fr 1fr')
    u_i = widgets.GridBox(widglist, layout=layout)
    out = widgets.interactive_output(func, widg)
    display(u_i, out)
    return seg.params
