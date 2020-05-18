#Full HPC Coco Run
import argparse
import random
import logging
import sys
import glob
import os
from pathlib import Path

from skimage import color
import imageio

import see
from see import GeneticSearch
from see import Segmentors
import pathlib

from see import DataDownload as dd


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the see-Semgent algorithm')
    parser.add_argument('-B', '--BMCV', action='store_false', help="Use BMCV Data")
    parser.add_argument('-S', '--Sky', action='store_true', help="Use Sky Data")
    parser.add_argument('-C', '--Coco', action='store_true', help="Use Coco Data")
    parser.add_argument("-c", "--checkpoint", 
                        help="Starting Population", 
                        type=str, default="")
    parser.add_argument("-g", "--generations", 
                        help="Number of Generations to run in search", 
                        type=int, default=2)
    parser.add_argument("-s", "--seed", 
                        help="Random Seed", 
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
                        type=str, default="./Output/")

    #Parsing Inputs
    args = parser.parse_args()
    print(args)

    #Setting Log Level
    logging.basicConfig(stream=sys.stdout, level=logging.ERROR)
    #logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    #Create output folder if dosn't exist
    Path(args.outputfolder).mkdir(parents=True, exist_ok=True)

    imagefiles, maskfiles, outputfiles = [], [], []
    
    #Make List of all files

    if(args.BMCV):
        print("Adding BMCV Data")
        files = dd.getBMCVFolderLists(outputfolder=args.outputfolder)
        imagefiles += files[0]
        maskfiles += files[1]
        outputfiles += files[2]

    if(args.Coco):
        print("Adding Coco Data")
        files = dd.getCocoFolderLists(outputfolder=args.outputfolder)
        imagefiles += files[0]
        maskfiles += files[1]
        outputfiles += files[2]
        
    if(args.Sky):
        print("Adding Sky Data")
        files = dd.getSkyFolderLists(outputfolder=args.outputfolder)     
        imagefiles += files[0]
        maskfiles += files[1]
        outputfiles += files[2]

    if imagefiles == []:
        print("Error: No dataset specified")
    
    print(f"processing: {len(imagefiles)} files")
                    
    startfile = args.checkpoint
#    population = ''
#    #Create Segmentor
#    #startfile = f"{args.outputfolder}Population_checkpoint.txt"
#    startfile = None
#    if startfile:
#        if os.path.exists(args.checkpoint):
#            params = ''
#        else:
#            params = eval(args.checkpoint)
#            startfile = None
#    else:
#        params=''
#        startfile=None
#    print(f"Algorithm={params}")


    #Pick out this image and mask
    index = args.index
    imagefile = imagefiles[index]
    maskfile = maskfiles[index]
    # Load this image and mask
    print(imagefile)
    img = imageio.imread(imagefile)
    gmask = imageio.imread(maskfile)
    #Run random Search
    random.seed(args.seed)
    ee = GeneticSearch.Evolver(img, gmask, pop_size=args.pop)
    population = ee.run(args.generations, startfile=startfile, checkpoint=args.checkpoint, cp_freq=9999)
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

        fitness = Segmentors.FitnessFunction(mask,gmask)
        file.write(f"{fitness[0]} {maskname}\n")
        print(f"evaluating {imagename} --> {fitness[0]}")

    file.close() 





