"""Download common and publicly avaliable segmentation datasets along with mask images."""

##TODO remove import os in favor of pathlib
##TODO Path.mkdir(mode=0o777, parents=False, exist_ok=False


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

DefaultFolder='./'

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

def downloadKOMATSUNA(filenames= ['rgbd_plant.zip', 'rgbd_label.zip'],
                      folder = f'{DefaultFolder}KOMATSUNA/',
                      urls = ['http://limu.ait.kyushu-u.ac.jp/~agri/komatsuna/rgbd_plant.zip',
                              'http://limu.ait.kyushu-u.ac.jp/~agri/komatsuna/rgbd_label.zip'],
                      datafolder=DefaultFolder,
                      force=True):
    """The KOMATSUNA plant dataset is a multisegmentation dataset avaliable at http://limu.ait.kyushu-u.ac.jp/~agri/komatsuna/"""
    
    if not os.path.exists(folder):
        os.makedirs(folder)
        print("Directory " , folder ,  " Created ")
    else:    
        print("Directory " , folder ,  " already exists")    
    
    for filename,url in zip(filenames, urls):
        zfile = Path(folder+filename)
        if not zfile.is_file() or force:
            print(f"Downloading {filename} from {url}")
            urlretrieve(url,folder+filename)
        else:
            print(f"File {filename} already exists")

        print(f"Unzipping {filename}")
        with zipfile.ZipFile(folder+filename, 'r') as zip_ref:
            zip_ref.extractall(folder)

        print(f"Download and Convert of {filename} Complete")

def downloadSky(filename = 'sky.zip', 
                    folder = DefaultFolder, 
                    url = 'https://www.ime.usp.br/~eduardob/datasets/sky/sky.zip',
                    force=True):
    """The sky dataset is a binary dataset avaliable at https://www.ime.usp.br/~eduardob/datasets/sky/"""
    
    if not os.path.exists(folder):
        os.makedirs(folder)
        print("Directory " , folder ,  " Created ")
    else:    
        print("Directory " , folder ,  " already exists")    
    
    zfile = Path(folder+filename)
    if not zfile.is_file() or force:
        print(f"Downloading {filename} from {url}")
        urlretrieve(url,folder+filename)

    print(f"Unzipping {filename}")
    with zipfile.ZipFile(folder+filename, 'r') as zip_ref:
        zip_ref.extractall(folder)
    
    print(f"Converting files in {folder}")
    images, masks, outputs = getSkyFolderLists()
    
    print(images)
    for i in masks:
        print(f"{i}")
        img = readpgm(i)
        img.astype(np.uint8)
        imageio.imsave(i,img)
        
    print(f"Download and Convert Complete")

def downloadCOSKEL(filename= 'SKEL_v1.1.zip',
                   folder = f'{DefaultFolder}',
                   url = 'https://github.com/jkoteswarrao/Object-Co-skeletonization-with-Co-segmentation/raw/master/CO-SKEL_v1.1.zip',
                   datafolder=DefaultFolder,
                   force=True):
    if not os.path.exists(folder):
        os.makedirs(folder)
        print("Directory " , folder ,  " Created ")
    else:
        print("Directory " , folder ,  " already exists")

    zfile = Path(folder+filename)
    if not zfile.is_file() or force:
        print(f"Downloading {filename} from {url}")
        urlretrieve(url,folder+filename)

    print(f"Unzipping {filename}")
    with zipfile.ZipFile(folder+filename, 'r') as zip_ref:
        zip_ref.extractall(folder)

    print(f"Download and Convert Complete")
    
    
def getSkyFolderLists(outputfolder='', folder=DefaultFolder):
    '''The Sky data has some odd filenames. This figures it out and creates
    Three lists for image, mask and output data.'''
    imagefolder = f"{folder}/sky/data/"
    maskfolder = f"{folder}/sky/groundtruth/"

    #print(f"{imagefolder} {maskfolder}")
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
        #print(f"{label}")
    return imagenames, masknames, outputnames

def getKomatsunaFolderLists(outputfolder='', folder=DefaultFolder):
    '''This downloads the KOMATSUNA dataset.'''

    imagefolder = f"{folder}/KOMATSUNA/multi_plant/"
    maskfolder = f"{folder}/KOMATSUNA/multi_label/"

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
        #print(f"{label}")
    return imagenames, masknames, outputnames

def getCOSKELFolderlists(outputfolder='output/', folder=DefaultFolder):
    '''The Sky data has some odd filenames. This figures it out and creates
    Three lists for image, mask and output data.'''
    imagefolder = Path(f"{folder}/CO-SKEL_v1.1/images/")
    maskfolder = Path(f"{folder}/CO-SKEL_v1.1/GT_masks/")

    imagePATHnames = list(Path(f'{imagefolder}').rglob('*.jpg'));
    imagenames = []
    masknames = []
    outputnames = []
    for index, file in enumerate(imagePATHnames):
        imagenames.append(str(file))
        #print(str(file))
        #print(imagefolder)
        filename = str(file).replace(str(imagefolder), '')
        name = filename[:-4]
        masknames.append(f"{maskfolder}{name}.png")
        outputnames.append(f"{outputfolder}{name}.png")
        #print(f"{filename}")

    return imagenames, masknames, outputnames
    
def getBMCVFolderLists(outputfolder=''):
    pth = pathlib.Path(__file__).parent.absolute()
    imagefolder = str(pth)+"/../Image_data/BMCV/"
    maskfolder = str(pth)+"/../Image_data/BMCV/"

    imagenames = []
    masknames = []
    outputnames = []

    imagenames.append(f'{imagefolder}/rgb_04_009_05.png')
    masknames.append(f'{imagefolder}/label_04_009_05.png')
    outputnames.append(f'{outputfolder}/label_04_009_05.png')

    imagenames.append(f'{imagefolder}/rgb_04_009_05.png')
    masknames.append(f'{imagefolder}/label_04_009_05299.png')
    outputnames.append(f'{outputfolder}/label_04_009_05299.png')

    imagenames.append(f'{imagefolder}/0020.jpg')
    masknames.append(f'{imagefolder}/0020_gt.pgm')
    outputnames.append(f'{outputfolder}/0020_gt.pgm')

    return imagenames, masknames, outputnames


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Download Image Data')
    parser.add_argument('-f', '--folder', default='./Image_data', help="Image data folder name")
    parser.add_argument('-P', '--Plant', action='store_false', help="Use Sky Data")
    parser.add_argument('-S', '--Sky', action='store_false', help="Use Sky Data")
    parser.add_argument('-C', '--COSKEL', action='store_false', help="Use COSKEL Data")
   
    #Parsing Inputs
    args = parser.parse_args()
    print(args)
    
    DefaultFolder = args.folder
    
    if args.Plant:
        downloadKOMATSUNA(force=False)

    if args.Sky:
        downloadSky(force=False)
    
    if args.COSKEL:
        downloadCOSKEL(force=False)
