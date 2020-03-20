from classes import ImageData
from classes import AlgorithmSpace
from classes.AlgorithmSpace import AlgorithmSpace
from classes import AlgorithmParams

from classes import FileClass
from classes.FileClass import FileClass
from classes import GeneticHelp
from classes.GeneticHelp import GeneticHelp as GA

from classes import RunClass
from classes.RunClass import RunClass as RC
from classes import RandomHelp
from classes.RandomHelp import RandomHelp as RandHelp

import threading
import multiprocessing
from multiprocessing import Process, Queue
import logging
import time
import sys
from sys import stderr, stdin, stdout
import os
import random
import copy
import pickle

IMAGE_PATH = 'Image_data\\Coco_2017_unlabeled\\rgbd_plant'
VALIDATION_PATH = 'Image_data\\Coco_2017_unlabeled\\rgbd_new_label'

def threadImg(imgPath, img, valImg, ):
	runner = RC.RC(imgPath, img, valImg)

	final = runner.RunGA()
	print(final[1])
	return final

def testStuff():
	print("Made a thread!!")
	return

#Need to incorporate threading
def background():
	while True:
		time.sleep(3)

def save_state():
	print("State saved")

if __name__=='__main__':
	#To determine the seed for debugging purposes
	seed = random.randrange(sys.maxsize)
	rng = random.Random(seed)
	print("Seed was:", seed)
	#seed = SEED

	#Will later have user input to find where the images are
	#Checking the directories
	if (FileClass.check_dir(IMAGE_PATH) == False):
		print ('ERROR: Directory \"%s\" does not exist'%IMAGE_PATH)
		sys.exit(1)

	if(FileClass.check_dir(VALIDATION_PATH) == False):
		print("ERROR: Directory \"%s\" does not exist"%VALIDATION_PATH)
		sys.exit(1)

	#Making an ImageData object for all of the regular images
	AllImages = [ImageData.ImageData(os.path.join(root, name)) for 
		root, dirs, files in os.walk(IMAGE_PATH) for name in files]


	#Making an ImageData object for all of the labeled images
	ValImages = [ImageData.ImageData(os.path.join(root, name)) for
		root, dirs, files in os.walk(VALIDATION_PATH) for name in
		files]

	imageCounter = 0
	goodAlgoCounter = 0
	goodAlgo = 0

	goodEnough = False
	initImg = 'Image_data\\Coco_2017_unlabeled\\rgbd_plant\\rgb_00_000_00.png'
	valImg = 'Image_data\\Coco_2017_unlabeled\\rgbd_new_label\\label_00_000_000.png'
	thread = threading.Thread(target=threadImg, args=IMAGE_PATH, )
	sys.exit()
	#Let's do some threading
	format = "%(asctime)s: %(message)s"
	#logging.basicConfig(fomat=format, level=logging.INFO, 
		#datefmt="%H:%M:%S")

	'''numFiles = input("Input the total number of files in your dataset ")
	print(numFiles)
	processes = multiprocessing.Queue(int(numFiles))
	p = multiprocessing.Process(target=time.sleep, args=(1000,))
	print(multiprocessing.current_process())
	#print(p, p.is_alive())
	#Do some kind of locking??
	for i in range(0,int(numFiles)):
		p = multiprocessing.Process(target=testStuff)
		processes.put(p,False)
		p.start()
		print ("Size of queue: ",processes.qsize())

	print(multiprocessing.active_children())
	cursor = 0
	'''
	print ("Possible commands are:\n\thelp -- Recreates this menu\n\t"
		   + "")
	while cursor < len(AllImages):
		#First we get the correct file
		isImg, isVal, imgFile, valFile = False, False, "", ""
		while(isImg == False and isVal == False):
			imgFile = input("Give the path to a image file in your dataset. ")
			valFile = input("Give the path to the segmented image of the last file. ")
		
			isImg = FileClass.check_dir(imgFile)
			isVal = FileClass.check_dir(valFile)
			if (isImg == False):
				print("Image file %s does not exist.\n"%imgFile)
			if (isVal == False):
				print("Segmentented image file %s does not exist.\n"%valFile)

		print("Good files")


		cursor += 1
		#If it's empty, we ignore it
		
		#Let's parse the line



		if command == 'image':
			#ImageData objects
			imageObj = ImageData.ImageData(imageFile)
			valObj = ImageData.ImageData(imageVal)
			#Let's write to the same image type that was read in
			imgType = findImageType(imageFile)
			imageName = "data\\newImage" + str(imageCounter) + imgType
			p = multiprocessing.Process(target=RC.RunGA, args=[AllVals, 
							SIGMA_MIN, SIGMA_MAX, SIGMA_WEIGHT, ITER,
							SMOOTH_MIN, SMOOTH_MAX, SMOOTH_WEIGHT, 
				  			BALLOON_MIN, BALLOON_MAX, BALLOON_WEIGHT,
				  			imgObj,valObj, imageName])
			processes.put(p, False())
			p.start()
			'''Algo, didWork = RunGA(AllVals, SIGMA_MIN, SIGMA_MAX, 
							SIGMA_WEIGHT, ITER, SMOOTH_MIN, 
							SMOOTH_MAX, SMOOTH_WEIGHT, 
				  			BALLOON_MIN, BALLOON_MAX, 
				  			BALLOON_WEIGHT, imgObj,valObj, 
				  			imageName)
			'''
			if (didWork == False):
				print("Did not converge for the algorithm")
				if goodAlgoCounter != 0:
					goodAlgoCounter -= 1
				continue
			else:
				#We found a good algorithm, We should test it on
				#another image as well
				goodAlgoCounter += 1
				goodAlgo = Algo

		break

		'''elif command == 'quit':
			#Should also get best image
			sys.exit()

		elif command == 'help':
			continue

		#if goodAlgoCounter >= 2:
		'''

	#Let's do some threading