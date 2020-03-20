import see

from skimage import color
import time
import imageio
import sys

from see import GeneticSearch

class SegmentImage():
    IMAGE_PATH = 'Image_data/Coco_2017_unlabeled//rgbd_plant'
    GROUNDTRUTH_PATH = 'Image_data/Coco_2017_unlabeled/rgbd_new_label'
    SEED = 134
    POPULATION = 10
    GENERATIONS = 2
    MUTATION = 0
    FLIPPROB = 0
    CROSSOVER = 0
    pop = None
    fitnesses = []

    VALIDATION_PATH="" # Not sure how this is used    
    def __init__(self, argv=[]):
        if argv:
            self.parseinput(argv)
    
    """Function to parse the command line inputs"""
    def parseinput(self, argv):   
        # The input arguments are Seed, population, generations, mutation, flipprob, crossover
        print("Parsing Inputs")

        self.SEED = int(argv[1])
        self.MUTATION = float(argv[3])
        self.FLIPPROB = float()

        try:
            self.SEED = int(argv[1])
        except ValueError:
            print("Incorrect SEED value, please input an integer")
        try:
            self.POPULATION = int(argv[2])
            assert self.POPULATION > 0
        except ValueError:
            print("Incorrect POPULATION value: Please input a positive integer.")
            sys.exit(2)
        except AssertionError:
            print("Incorrect POPULATION value: Please input a positve integer.")
            sys.exit(2)

        try:
            self.GENERATIONS = int(argv[3])
        except ValueError:
            print("Incorrect value for GENERATIONS. Please input a positive integer.")
            sys.exit(2)
        except AssertionError:
            print("Incorrect value for GENERATIONS. Please input a positive integer.")
            sys.exit(2)

        try:
            self.MUTATION = float(argv[4])
            assert 0 <= self.MUTATION <= 1

        except ValueError:
            print("Please make sure that MUTATION is a positive percentage (decimal).")
            sys.exit(2)
        except AssertionError:
            print("Please make sure that MUTATION is a positive percentage (decimal).")
            sys.exit(2)

        try:
            self.FLIPPROB = float(argv[5])
            assert 0 <= self.FLIPPROB <= 1
        except ValueError:
            print("Incorrect value for FLIPPROB. Please input a positive percentage (decimal).")
            sys.exit(2)
        except AssertionError:
            print("Incorrect value for FLIPPROB. Please input a positive percentage (decimal).")
            sys.exit(2)

        try:
            self.CROSSOVER = float(argv[6])
            assert 0 <= self.CROSSOVER <= 1
        except ValueError:
            print(
                "Incorrect value for CROSSOVER. Please input a positive percentage (decimal).")
            sys.exit(2)
        except AssertionError:
            print(
                "Incorrect value for CROSSOVER. Please input a positive percentage (decimal).")
            sys.exit(2)


#         # Checking the directories
#         if (FileClass.check_dir(self.IMAGE_PATH) == False):
#             print('ERROR: Directory \"%s\" does not exist' % self.IMAGE_PATH)
#             sys.exit(1)

#         if(FileClass.check_dir(self.GROUNDTRUTH_PATH) == False):
#             print("ERROR: Directory \"%s\" does not exist" % self.VALIDATION_PATH)
#             sys.exit(1)

        return 

    # TODO rewrite "main" as part of a class or function structure.
    # TODO rewrite to make it pleasently parallel.
    """Function to run the main GA search function"""
    def runsearch(self):
        img = imageio.imread('Image_data/Coco_2017_unlabeled//rgbd_plant/rgb_00_000_00.png')
        gmask = imageio.imread('Image_data/Coco_2017_unlabeled/rgbd_new_label/label_00_000_000.png')
        if len(gmask.shape) > 2:
            gmask = color.rgb2gray(gmask)
        ee = GeneticSearch.Evolver(img, gmask)
        #ee.run(self.GENERATIONS)
        population = ee.run(self.GENERATIONS, startfile="0_checkpoint.json", checkpoint="2_checkpoint.json")

if __name__ == '__main__':
    ga = SegmentImage(sys.argv)
    ga.runsearch()
