# Python version 3.6.8
# Keras version 2.4.3
# Tensorflow version 2.4.1
# OpenCV version 4.5.1

import cv2
import numpy as np
import pickle

def build_squares(img):
	x, y, w, h = 420, 140, 10, 10
	d = 10
	imgCrop = None
	crop = None
	# 10 X 5 grid of squares.
	for i in range(10):
		for j in range(5):
			if np.any(imgCrop == None):
				# The cropped portion of the image for the first time.
				imgCrop = img[y:y+h, x:x+w]
			else:
				# Stack horizontally for subsequent images.
				imgCrop = np.hstack((imgCrop, img[y:y+h, x:x+w]))
			# Draw a rectange on the image.
			cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 1)
			x+=w+d
		# Stack vertically as well.
		if np.any(crop == None):
			crop = imgCrop
		else:
			crop = np.vstack((crop, imgCrop)) 
		imgCrop = None
		x = 420
		y+=h+d
	# Return the final stacked images.
	return crop

def get_hand_hist():
	# Start capturing from device 1.
	# The camera object is stored here.
	cam = cv2.VideoCapture(1)
	# If unable to read, choose device 0
	if cam.read()[0]==False:
		cam = cv2.VideoCapture(0)
	x, y, w, h = 300, 100, 300, 300
	flagPressedC, flagPressedS = False, False
	imgCrop = None
	while True:
		# Begin reading the frames.
		# Two values are returned, the status, and the actual frame.
		img = cam.read()[1]
		# Flip the image along the Y-axis-Mirror image.
		img = cv2.flip(img, 1)
		# Resize the image to match our resolution.
		img = cv2.resize(img, (640, 480))
		# Change the color scheme of the image to hue-saturation-values.
		hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
		
		#Wait for a keypress event for 1 ms.
		keypress = cv2.waitKey(1)
		
		#If it is 'c',capture it.
		if keypress == ord('c'):
		# Convert the interested portion of the image and make a histogram out of it.		
			hsvCrop = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2HSV)
			flagPressedC = True
			# Create the histogram.
			hist = cv2.calcHist([hsvCrop], [0, 1], None, [180, 256], [0, 180, 0, 256])
			# Normalize it.
			cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
		#If it is 's',save it.
		elif keypress == ord('s'):
			flagPressedS = True	
			break
		if flagPressedC:
			# Histogram backprojection for feature detection.	
			dst = cv2.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 1)
			dst1 = dst.copy()
			# Apply a structuring element.
			# An ellipse is chosen as it highlights the central portion of the image.
			disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
			cv2.filter2D(dst,-1,disc,dst)
			# Apply Gaussian and Median blur
			blur = cv2.GaussianBlur(dst, (11,11), 0)
			blur = cv2.medianBlur(blur, 15)
			ret,thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
			# Merge everything
			thresh = cv2.merge((thresh,thresh,thresh))
			#cv2.imshow("res", res)
			cv2.imshow("Thresh", thresh)
		if not flagPressedS:
			#Get the cropped image
			imgCrop = build_squares(img)
		#cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
		cv2.imshow("Set hand histogram", img)
	# Close the camera
	cam.release()
	#Clear the screen and save the histogram
	cv2.destroyAllWindows()
	with open("hist", "wb") as f:
		pickle.dump(hist,f)

get_hand_hist()

