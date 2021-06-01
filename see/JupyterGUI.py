"""Produces a GUI.

Allows users to switch between segmentation
algorithms and alter the parameters manually using a slider. It shows two images,
one with the original image with the resulting mask and one with the original image
with the negative of the resulting mask.
"""

import matplotlib.pylab as plt
import ipywidgets as widgets
from IPython.display import display, clear_output
from pathlib import Path
from see.Segmentors import segmentor, seg_params
from see.ColorSpace import colorspace
import imageio
import numpy as np

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
    """Choose image from available set of images."""
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
        print(f"img = imageio.imread(\'{w.value}\')")
        print(f"gmask = imageio.imread(\'{masklist[index]}\')")
        
    def on_change(change):
        if change['type'] == 'change' and change['name'] == 'value':
            update(w)

    w.observe(on_change)
    update(w)
    return w



def picksegment(algorithms):
    """Provide capabilities for user to choose a segmentation."""
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


#TODO: add more color representations in the output
"""
'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'winter', 'winter_r'
"""
"""https://matplotlib.org/stable/tutorials/colors/colormaps.html"""
    
def showimage(img, ax=None, color='RGB', multichannel=True, channel=2):
    """Display image as part of GUI."""
    if not ax:
        fig = plt.figure()
        ax=fig.gca()
        
    if len(img.shape) == 2:
        multichannel=False
    
    if multichannel:    
        if color=='RGB':
            ax.imshow(img)
            return
        elif color=='‘HSV':
            pass
        elif color=='RGB CIE':
            pass        
        elif color=='XYZ':
            pass        
        elif color=='YUV':
            pass
        elif color=='YIQ':
            pass
        elif color=='YPbPr':
            pass
        elif color=='YCbCr':
            pass
        elif color=='YDbDr':
            pass   
        ax.imshow(img,cmap='gray')
    else:
        print(f'singlechannel {color} {channel}')
        if len(img.shape) == 2: 
            single_channel=img
        else:
            single_channel=img[:,:,channel]
        
        if color=='RGB':
            c_im = np.ubyte(np.zeros([img.shape[0], img.shape[1], 3]))
            c_im[:,:,channel] = single_channel;
            ax.imshow(c_im);
            return
        elif color=='HSV':
            print('HSV')
            if channel==0:
                ax.imshow(single_channel,cmap='hsv')
                return
            elif channel==2:
                ax.imshow(single_channel,cmap='gray')
                return
        elif color=='RGB CIE':
            pass        
        elif color=='XYZ':
            pass        
        elif color=='YUV':
            pass
        elif color=='YIQ':
            pass
        elif color=='YPbPr':
            pass
        elif color=='YCbCr':
            pass
        elif color=='YDbDr':
            pass  
        ax.imshow(single_channel)
    
def colorwidget(img, paramlist=None):
    """Display info to user on colorspace if paramlist is not empty."""
    if paramlist:
        seg = colorspace(paramlist=paramlist)
    else:
        seg = colorspace()

    print(f"seg.params = {seg.params}")
    widg = dict()
    widglist = []

    for ppp, ind in zip(seg.paramindexes, range(len(seg.paramindexes))):
        thislist = seg.params.ranges[ppp]
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
        for k in kwargs:
            seg.params[k] = kwargs[k]
        colorspace = seg.evaluate(img)
        
        #fit = Segmentors.FitnessFunction(mask, gmask)
        fig = showtwo(img, colorspace)
        showimage(img, ax=fig.gca(), 
                  color=seg.params['colorspace'], 
                  multichannel=seg.params['multichannel'], 
                  channel=seg.params['channel'])
        # I like the idea of printing the sharepython but it should be below the figures. 
        #print(seg.sharepython(img))
#         plt.title('Fitness Value: ' + str(fit[0]))

    
    layout = widgets.Layout(grid_template_columns='1fr 1fr 1fr')
    u_i = widgets.GridBox(widglist, layout=layout)
    out = widgets.interactive_output(func, widg)
    display(u_i, out)
    
    return seg.params    

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
        thislist = seg.params.ranges[ppp]
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
