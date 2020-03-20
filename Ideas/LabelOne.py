#Change the labels from tri-color (or more) to binary

import cv2
from PIL import Image
import os

IMAGE = "label_00_000_00.png"
IMAGE_PATH = "Image_data\\Coco_2017_unlabeled\\rgbd_label"
OUT_PATH = "Image_data\\Coco_2017_unlabeled\\rgbd_new_label"
if __name__ == '__main__':
	
	#Puts all of the paths to the images to 
	AllImages = [os.path.join(root, name) for 
		root, dirs, files in os.walk(IMAGE_PATH) for name in files]
	AllNames = [name for root, dirs, files in os.walk(IMAGE_PATH) 
		for name in files]
	counter = 0
	for image in AllImages:
		picture = Image.open(image, mode='r')
		width = picture.width
		height = picture.height

		for x in range(0, width):
			for y in range(0,height):
				pixel = picture.getpixel((x,y))
				if pixel[0] == 0 and pixel[1] == 0 and pixel[2] == 0:
					continue
				else:
					picture.putpixel((x, y), 134)
		picture.convert(mode='L')
		imgName = OUT_PATH + "\\" + AllNames[counter].strip(".png") + str(counter) + ".png"
		print(imgName)
		picture.save(imgName)

		counter += 1




	'''for x in picDim[0]:
		for y in picDim[1]:
			color = picture.getpixel((x,y))
			print(color)
	'''
	#cv2.imwrite("fileWrite.png", file)

