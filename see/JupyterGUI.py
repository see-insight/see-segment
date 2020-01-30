
import matplotlib.pylab as plt
from ipywidgets import interact
import ipywidgets as widgets

from see import Segmentors

def showtwo(img, img2):
    fig = plt.figure(figsize=(20,20))
    ax = fig.add_subplot(1,2,1)
    ax.imshow(img)
    ax = fig.add_subplot(1,2,2)
    ax.imshow(img2)
    
def showthree(im, img, img2):
    fig = plt.figure(figsize=(20,20))
    ax = fig.add_subplot(1,3,1)
    ax.imshow(im)
    ax = fig.add_subplot(1,3,2)
    ax.imshow(img)
    ax = fig.add_subplot(1,3,3)
    ax.imshow(img2)
    
def showSegment(im, mask):
    im1 = im.copy()
    im2 = im.copy()
    im1[mask>0,:] = 0
    im2[mask==0,:] = 0
    showtwo(im1,im2)


def segmentwidget(params, img, gmask):
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
        print(seg.params["algorithm"])
        for k in kwargs:
            seg.params[k] = kwargs[k]
        mask = seg.evaluate(img)
        fit = Segmentors.FitnessFunction(mask,gmask)
        showSegment(img,mask)
        plt.title(fit)
        

    layout = widgets.Layout(grid_template_columns='1fr 1fr 1fr')
    ui = widgets.GridBox(widglist, layout=layout)

    out = widgets.interactive_output(f, widg)
    display(ui, out)

    
    