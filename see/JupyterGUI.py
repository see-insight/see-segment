"""This produces a GUI that allows users to switch between segmentation
 algorithms and alter the parameters manually using a slider. It shows two images,
  one with the original image with the resulting mask and one with the original image
   with the negative of the resulting mask."""
import matplotlib.pylab as plt
import ipywidgets as widgets
from IPython.display import display, clear_output
from pathlib import Path
from see.Segmentors import segmentor, seg_params
import imageio


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

def pickimage(folder='Image_data/Examples/'):
    #def pickimage(

    directory = Path(folder)

    allfiles = sorted(directory.glob('*'))

    filelist = []
    masklist = []
    for file in allfiles:
        if file.suffix ==".jpg" or file.suffix ==".jpeg" or file.suffix ==".JPEG" or file.suffix ==".png":
            if not "_GT" in file.name:
                filelist.append(file)
                mask = directory.glob(f"{file.stem}_GT*")
                for m in mask:
                    masklist.append(m)
    
    w = widgets.Dropdown(
        options=filelist,
        value=filelist[0],
        description='Choose image:',
    )

    def update(w):
        clear_output(wait=True) # Clear output for dynamic display
        display(w)
        w.img = imageio.imread(w.value)
        index = filelist.index(w.value)
        w.gmask = imageio.imread(masklist[index])
        if len(w.gmask.shape) > 2:
            w.gmask = w.gmask[:,:,0]
        fig = showtwo(w.img, w.gmask)
        print(f"import imageio")
        print(f"data.img = imageio.imread(\'{w.value}\')")
        print(f"data.gmask = imageio.imread(\'{masklist[index]}\')")
        
    def on_change(change):
        if change['type'] == 'change' and change['name'] == 'value':

            update(w)

    w.observe(on_change)
    update(w)
    return w


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
            print(segmentor.algorithmspace[change['new']].__doc__)
            print(f"\nsegmentor_name=\'{w.value}\'")
    w.observe(on_change)

    display(w)
    print(segmentor.algorithmspace[w.value].__doc__)
    print(f"\nalg.value=\'{w.value}\'")
    return w

def segmentwidget(img, params=None, alg=None):
    """Generate GUI. Produce slider for each parameter for the current segmentor.
     Show both options for the masked image.

    Keyword arguments:
    img -- original image
    gmask -- ground truth segmentation mask for the image
    params -- list of parameter options
    alg -- algorithm to search parameters over

    """
    if params:
        if alg:
            params['algorithm'] = alg;
        seg = segmentor.algoFromParams(params)
    else:
        if alg:
            algorithm_gen = segmentor.algorithmspace[alg]
            seg = algorithm_gen()
        else:
            seg = segmentor()

    widg = dict()
    widglist = []

    for ppp, ind in zip(seg.paramindexes, range(len(seg.paramindexes))):
        thislist = seg_params.ranges[ppp]
        name = ppp
        current_value = seg.params[ppp]
        if not current_value in thislist:
            #TODO: We should find the min distance between current_value and this list and use that instead.
            current_value = thislist[0]
            
        thiswidg = widgets.SelectionSlider(options=tuple(thislist),
                                           disabled=False,
                                           description=name,
                                           value=current_value,
                                           continuous_update=False,
                                           orientation='horizontal',
                                           readout=True
                                          )

        widglist.append(thiswidg)
        widg[ppp] = thiswidg

    
    def func(**kwargs):
        """Find mask and fitness for current algorithm. Show masked image."""
        print(seg.params["algorithm"])
        for k in kwargs:
            seg.params[k] = kwargs[k]
        mask = seg.evaluate(img)
        #fit = Segmentors.FitnessFunction(mask, gmask)
        fig = showtwo(img, mask)
        # I like the idea of printing the sharepython but it should be below the figures. 
        #print(seg.sharepython(img))
#         plt.title('Fitness Value: ' + str(fit[0]))

    
    layout = widgets.Layout(grid_template_columns='1fr 1fr 1fr')
    u_i = widgets.GridBox(widglist, layout=layout)
    out = widgets.interactive_output(func, widg)
    display(u_i, out)
    
    return seg.params
