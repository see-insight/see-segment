#Changes a binary labeled image to black and white
from PIL import Image
import os

IMAGE = "label_00_000_00.png"
IMAGE_PATH = "Image_data\\Coco_2017_unlabeled\\rgbd_new_label"
OUT_PATH = "Image_data\\Coco_2017_unlabeled\\rgbd_another_label"


if __name__ == '__main__':

	AllImages = [os.path.join(root, name) for 
		root, dirs, files in os.walk(IMAGE_PATH) for name in files]
	AllNames = [name for root, dirs, files in os.walk(IMAGE_PATH) 
		for name in files]
	counter = 0
	for image in AllImages:
		colored = Image.open(image)
		gray = colored.convert('L')sa
		bw = gray.point(lambda x: 0 if x < 150 else 255, '1')

		imgName = OUT_PATH + "\\" + AllNames[counter].strip(".png") +str(counter) + ".png"
		bw.save(imgName)
		counter += 1


