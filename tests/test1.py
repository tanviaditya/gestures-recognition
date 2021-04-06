import cv2
import pickle

#My web camera.
webcamera=cv2.VideoCapture(0)

try:
	# run a infinite loop 
	while(True):
		# read a video frame by frame
		# read() returns tuple in which 1st item is boolean value 
		# either True or False and 2nd item is frame of the video.
		# read() returns False when live video is ended so 
		# no frame is readed and error will be generated.
		ret,frame=webcamera.read()
		





		# show the frame on the newly created image window
		cv2.imshow('Frames',frame)
		# this condition is used to run a frames at the interval of 10 mili sec
		# and if in b/w the frame running , any one want to stop the execution.
		key=cv2.waitKey(10)
		if key==ord('q'):break
except Exception as E:
	print(E)
	print("Video has ended..")
