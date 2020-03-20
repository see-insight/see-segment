import argparse
import random
import logging
import sys

from skimage import color
import imageio

import see
from see import GeneticSearch
from see import Segmentors

parser = argparse.ArgumentParser(description='Run the see-Semgent algorithm')
parser.add_argument("-g", "--generations", 
                    help="Number of Generations to run in search", 
                    type=int, default=10)
parser.add_argument("-s", "--seed", 
                    help="Random Seed", 
                    type=int, default=0)
parser.add_argument("-p", "--pop", 
                    help="Population (file or number)", 
                    type=int, default="100")
parser.add_argument("-i", "--image", 
                    help="Input image file", 
                    type=str, 
                    default="Image_data/Coco_2017_unlabeled//rgbd_plant/rgb_00_000_00.png")
parser.add_argument("-m", "--mask", 
                    help="Mask ground truth", 
                    type=str, 
                    default="Image_data/Coco_2017_unlabeled/rgbd_new_label/label_00_000_000.png")
parser.add_argument("-c", "--checkpointfile", 
                    help="Rootname for checkpoint file", 
                    type=str, default="")
parser.add_argument("-o", "--outputfolder", 
                    help="Output Folder", 
                    type=str, default="")

args = parser.parse_args()
print(args)

random.seed(args.seed)

logging.basicConfig(stream=sys.stdout, level=logging.ERROR)
#logging.basicConfig(stream=sys.stdout, level=logging.INFO)

img = imageio.imread(args.image)
gmask = imageio.imread(args.mask)

if len(gmask.shape) > 2:
    gmask = color.rgb2gray(gmask)

ee = GeneticSearch.Evolver(img, gmask, pop_size=args.pop)
ee.run(args.generations, checkpoint=args.checkpointfile)

seg = Segmentors.algoFromParams(ee.hof[0])
mask = seg.evaluate(img)

imageio.imwrite(args.outputfolder+"file.jpg", mask)

fitness,_ = Segmentors.FitnessFunction(mask,gmask)
print(f"{fitness} {ee.hof[0]}")





