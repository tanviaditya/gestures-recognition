# Python version 3.6.8
# Keras version 2.4.3
# Tensorflow version 2.4.1
# OpenCV version 4.5.1
import cv2, os

def flip_images():
	gest_folder = "gestures"
	images_labels = []
	images = []
	labels = []
	for g_id in os.listdir(gest_folder):
		for i in range(600):
			path = gest_folder+"/"+g_id+"/"+str(i+1)+".jpg"
			new_path = gest_folder+"/"+g_id+"/"+str(i+1+10)+".jpg"
#			print(path)
			img = cv2.imread(path, 0)
			img = cv2.flip(img, 1)
#			print(new_path)
			cv2.imwrite(new_path, img)

flip_images()
