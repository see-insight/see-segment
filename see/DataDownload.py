import argparse
import random
import logging
import sys
import glob
import os
from pathlib import Path
from urllib.request import urlretrieve
import zipfile

from skimage import color
import imageio
import pathlib

import numpy as np
import matplotlib.pyplot as plt

def readpgm(name):
    """The ground truth data is in ascii P2 pgm binary files.  
    OpenCV can read these files in but it would be much easier 
    to just convert them to more common pgm binary format (P5)."""
    with open(name, encoding="utf8", errors='ignore') as f:
        lines = f.readlines()

    # Ignores commented lines
    for l in list(lines):
        if l[0] == '#':
            lines.remove(l)

    # Makes sure it is ASCII format (P2)
    filetype = lines[0].strip()
    print(f"Filetype is {filetype}")
    if filetype == 'P2':
        print('Trying to read as P2 PGM file')
        # Converts data to a list of integers
        data = []
        for line in lines[1:]:
            data.extend([int(c) for c in line.split()])
        img = np.reshape(np.array(data[3:]),(data[1],data[0]))
    else:
        print('Trying to read as P5 PGM file')
        img = imageio.imread(name)
    return img

def downloadKOMATSUNA(filenames= ['multi_label.zip', 'multi_plant.zip'],
                      folder = 'KOMATSUNA/',
                      urls = ['http://limu.ait.kyushu-u.ac.jp/~agri/komatsuna/multi_plant.zip',
                              'http://limu.ait.kyushu-u.ac.jp/~agri/komatsuna/multi_label.zip'],
                      force=True):

    ##TODO## Make folder
    for filename,url in zip(filenames, urls):
        zfile = Path(folder+filename)
        if not zfile.is_file() or force:
            print(f"Downloading {filename} from {url}")
            urlretrieve(url,folder+filename)

        print(f"Unzipping {filename}")
        with zipfile.ZipFile(folder+filename, 'r') as zip_ref:
            zip_ref.extractall(folder)

        print(f"Download and Convert Complete")

def downloadSkyData(filename = 'sky.zip', 
                    folder = '.', 
                    url = 'https://www.ime.usp.br/~eduardob/datasets/sky/sky.zip',
                    force=True):
    
    zfile = Path(folder+filename)
    if not zfile.is_file() or force:
        print(f"Downloading {filename} from {url}")
        urlretrieve(url,folder+filename)

    print(f"Unzipping {filename}")
    with zipfile.ZipFile(folder+filename, 'r') as zip_ref:
        zip_ref.extractall(folder)
    
    print(f"Converting files in {folder}")
    images, masks, outputs = getSkyFolderLists()
    for i in masks:
        print(f"{i}")
        img = readpgm(i)
        img.astype(np.uint8)
        imageio.imsave(i,img)
        
    print(f"Download and Convert Complete")

def getSkyFolderLists(outputfolder=''):
    '''The Sky data has some odd filenames. This figures it out and creates
    Three lists for image, mask and output data.'''
    pth = pathlib.Path(__file__).parent.absolute()
    imagefolder = str(pth)+"../Image_data/sky/data/"
    maskfolder = str(pth)+"../Image_data/sky/groundtruth/"

    imagenames = glob.glob(f'{imagefolder}/*.jpg')
    imagenames.sort()
    masknames = []
    outputnames = []
    for index, name in enumerate(imagenames):
        imagename = os.path.basename(name)
        image_id = imagename[:-4]
        label = f"{image_id}_gt.pgm"
        masknames.append(f"{maskfolder}/{label}")
        outputnames.append(f"{outputfolder}{label}")
    return imagenames, masknames, outputnames

def getKomatsunaFolderLists(outputfolder=''):
    '''The Coco data has some odd filenames. This figures it out and creates
    Three lists for image, mask and output data.'''
    pth = pathlib.Path(__file__).parent.absolute()
    imagefolder = str(pth)+"../Image_data/KOMATSUNA/multi_plant/"
    maskfolder = str(pth)+"../Image_data/KOMATSUNA/multi_label/"

    imagenames = glob.glob(f'{imagefolder}*.png')
    imagenames.sort()
    masknames = []
    outputnames = []
    for index, name in enumerate(imagenames):
        imagename = os.path.basename(name)
        image_id = imagename[4:-4]
        label = f"label_{image_id}.png"
        masknames.append(f"{maskfolder}{label}")
        outputnames.append(f"{outputfolder}{label}")
    return imagenames, masknames, outputnames

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Download Sky Data')
    parser.add_argument('-S', '--Sky', action='store_false', help="Use Sky Data")
    
    parser = argparse.ArgumentParser(description='Download |Plant Data')
    parser.add_argument('-P', '--Plant', action='store_false', help="Use Sky Data")

    #Parsing Inputs
    args = parser.parse_args()
    print(args)

    if args.Sky:
        downloadSkyData(force=False)
        
    if args.Plant
        downloadKOMATSUNA(force=False)
    
