#Full HPC Sky Run
import argparse
import random
import logging
import sys
import glob
import os
from pathlib import Path

from skimage import color
import imageio


import pathlib

import numpy as np
import matplotlib.pyplot as plt


def readpgm(name):
    """The ground truth data is in ascii P2 pgm binary files.  OpenCV can read these files in but it would be much easier to just convert them to more common pgm binary format (P5)."""
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

def downloadSkyData(filename = 'sky.zip', 
                    folder = '.', 
                    url = 'https://www.ime.usp.br/~eduardob/datasets/sky/sky.zip',
                    force=True):
    from urllib.request import urlretrieve
    import zipfile

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
    imagefolder = str(pth)+"/../Image_data/sky/data/"
    maskfolder = str(pth)+"/../Image_data/sky/groundtruth"

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

if __name__ == "__main__":
    import see
    from see import GeneticSearch
    from see import Segmentors
    parser = argparse.ArgumentParser(description='Run the see-Semgent algorithm')
    parser.add_argument("-c", "--checkpoint", 
                        help="Starting Population", 
                        type=str, default="")
    parser.add_argument("-g", "--generations", 
                        help="Number of Generations to run in search", 
                        type=int, default=2)
    parser.add_argument("-s", "--seed", 
                        type=int, default=0)
    parser.add_argument("-p", "--pop", 
                        help="Population (file or number)", 
                        type=int, default="10")
    parser.add_argument("-i", "--index", 
                        help="Input image index (used for looping)", 
                        type=int, 
                        default=0)
    parser.add_argument("-o", "--outputfolder", 
                        help="Output Folder", 
                        type=str, default="./output/")

    #Parsing Inputs
    args = parser.parse_args()
    print(args)

    #Setting Log Level
    logging.basicConfig(stream=sys.stdout, level=logging.ERROR)
    #logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    #Create output folder if dosn't exist
    Path(args.outputfolder).mkdir(parents=True, exist_ok=True)

    
    #Make List of all files
    imagefiles, maskfiles, outputfiles = getSkyFolderLists(outputfolder=args.outputfolder)

    population = ''
    #Create Segmentor
    #startfile = f"{args.outputfolder}Population_checkpoint.txt"
    startfile = None
    if startfile:
        if os.path.exists(args.checkpoint):
            params = ''
        else:
            params = eval(args.checkpoint)
            startfile = None
    else:
        params=''
        startfile=None
    print(f"Algorithm={params}")


    if params:
        #Check to see if list of parameters is passed
        if len(params[0])>1:
            #Pick this parameter from list
            if args.index:
                params = params[args.index]
    else:
        #Pick out this image and mask
        index = args.index
        imagefile = imagefiles[index]
        maskfile = maskfiles[index]

        # Load this image and mask
        img = imageio.imread(imagefile)
        gmask = imageio.imread(maskfile)

        #Run random Search
        random.seed(args.seed)
        ee = GeneticSearch.Evolver(img, gmask, pop_size=args.pop)
        population = ee.run(args.generations)#, startfile=startfile)# TODO: ADD THIS, checkpoint=args.checkpointfile)
        ee.writepop(population, filename=f"{args.outputfolder}Population_checkpoint.txt")
        params = ee.hof[0]

    #Create segmentor from params
    file = open(f"{args.outputfolder}params.txt","w") 
    file.write(str(params)+"\n") 

    seg = Segmentors.algoFromParams(params)

    #Loop though images
    for imagename, maskname, outputname in zip(imagefiles, maskfiles, outputfiles):

        # Loop over image files
        img = imageio.imread(imagename)
        gmask = imageio.imread(maskname)
        if len(gmask.shape) > 2:
            gmask = color.rgb2gray(gmask)

        #Evaluate image
        mask = seg.evaluate(img)

        #Save Mask to output
        print(f"writing {outputname}")
        imageio.imwrite(outputname, mask)

        fitness,_ = Segmentors.FitnessFunction(mask,gmask)
        file.write(f"{fitness} {maskname}\n")
        print(f"evaluating {imagename} --> {fitness}")

    file.close() 




