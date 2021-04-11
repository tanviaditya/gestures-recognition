# Python version 3.6.8
# Keras version 2.4.3
# Tensorflow version 2.4.1
# OpenCV version 4.5.1
# sklearn version 0.24.1


import cv2, pickle
import numpy as np
import tensorflow as tf
import smtplib 
import os
import sqlite3
import pyttsx3
from keras.models import load_model
from threading import Thread
from twilio.rest import Client
engine = pyttsx3.init()
engine.setProperty('rate', 150)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
model = load_model('cnn_model_keras2_new3.h5')

def get_hand_hist():
	with open("hist", "rb") as f:
		hist = pickle.load(f)
	return hist

def get_image_size():
	img = cv2.imread('gestures/0/10.jpg', 0)
	return img.shape

image_x, image_y = get_image_size()

def keras_process_image(img):
	img = cv2.resize(img, (image_x, image_y))
	img = np.array(img, dtype=np.float32)
	img = np.reshape(img, (1, image_x, image_y, 1))
	return img

def keras_predict(model, image):
	processed = keras_process_image(image)
	pred_probab = model.predict(processed)[0]
	pred_class = list(pred_probab).index(max(pred_probab))
	return max(pred_probab), pred_class

def get_pred_text_from_db(pred_class):
	conn = sqlite3.connect("gesture_db.db")
	cmd = "SELECT g_name FROM gesture WHERE g_id="+str(pred_class)
	cursor = conn.execute(cmd)
	for row in cursor:
		return row[0]

def get_pred_from_contour(contour, thresh):
	x1, y1, w1, h1 = cv2.boundingRect(contour)
	save_img = thresh[y1:y1+h1, x1:x1+w1]
	text = ""
	if w1 > h1:
		save_img = cv2.copyMakeBorder(save_img, int((w1-h1)/2) , int((w1-h1)/2) , 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
	elif h1 > w1:
		save_img = cv2.copyMakeBorder(save_img, 0, 0, int((h1-w1)/2) , int((h1-w1)/2) , cv2.BORDER_CONSTANT, (0, 0, 0))
	pred_probab, pred_class = keras_predict(model, save_img)
	print("Pred class ",pred_class)
	print("Pred prob ",pred_probab)
	if pred_probab*100 > 35:
		text = get_pred_text_from_db(pred_class)
		
	return text

hist = get_hand_hist()
x, y, w, h = 300, 100, 300, 300
# is_voice_on = True

def get_img_contour_thresh(img):
	img = cv2.flip(img, 1)
	imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	dst = cv2.calcBackProject([imgHSV], [0, 1], hist, [0, 180, 0, 256], 1)
	disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
	cv2.filter2D(dst,-1,disc,dst)
	blur = cv2.GaussianBlur(dst, (11,11), 0)
	blur = cv2.medianBlur(blur, 15)
	thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
	thresh = cv2.merge((thresh,thresh,thresh))
	thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
	thresh = thresh[y:y+h, x:x+w]
	contours = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
	return img, contours, thresh

# def say_text(text):
# 	if not is_voice_on:
# 		return
# 	while engine._inLoop:
# 		pass
# 	engine.say(text)
# 	engine.runAndWait()

def sendSms():
	number1='sender number'
	number2='reciever number'
	ACCOUNT_SID='account SID'
	AUTH_TOKEN='auth token'
	account_sid = ACCOUNT_SID
	auth_token = AUTH_TOKEN
	client = Client(account_sid, auth_token)
	message = client.messages \
    .create(
         body='SOS I need help',
         from_=number1,
         to=number2
    )

	print(message.sid)

def sendEmail():
	server = smtplib.SMTP_SSL('smtp.gmail.com', 465) 
	server.ehlo() 
	email='your_email@gmail.com'
	password='your_password'
	server.login(email, password) 
	print('Login done')
	semail='another_email@gmail.com'
	message='SOS I need help'
	server.sendmail(email,[semail], message) 

def text_mode(cam):
	# global is_voice_on
	text = ""
	word = ""
	count_same_frame = 0
	help_counter = 0
	while True:
		img = cam.read()[1]
		img = cv2.resize(img, (640, 480))
		img, contours, thresh = get_img_contour_thresh(img)
		old_text = text
		if len(contours) > 0:
			contour = max(contours, key = cv2.contourArea)
			if cv2.contourArea(contour) > 10000:
				text = get_pred_from_contour(contour, thresh)
				# print("Prediction ", text)
				if old_text == text:
					count_same_frame += 1
				else:
					count_same_frame = 0

				if count_same_frame > 20:
					word = word + text
					f = open("demofile.txt", "a")
					f.write(word+"\n")
					f.close()
					if(text=='help'):
						help_counter+=1
					if(help_counter==2):
						sendSms()
						sendEmail()
						help_counter=0
					count_same_frame = 0

			elif cv2.contourArea(contour) < 1000:
				if word != '':
					print('yolo')

				text = ""
				word = ""
		else:
			if word != '':
				print('yolo1')
			text = ""
			word = ""
		blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
		cv2.putText(blackboard, " ", (180, 50), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (255, 0,0))
		cv2.putText(blackboard, "Prediction - " + text, (30, 100), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 0))
		cv2.putText(blackboard, word, (30, 240), cv2.FONT_HERSHEY_TRIPLEX, 2, (255, 255, 255))
		cv2.putText(blackboard, " ", (450, 440), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 127, 0))
		cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
		res = np.hstack((img, blackboard))
		cv2.imshow("Recognizing gesture", res)
		cv2.imshow("thresh", thresh)
		keypress = cv2.waitKey(1)
		if keypress == ord('q'):
			# break
			return
		if keypress == ord('h'):
			sendSms()
			sendEmail()


def recognize():
	cam = cv2.VideoCapture(0)
	text = ""
	word = ""
	count_same_frame = 0
	keypress = 1
	text_mode(cam)

keras_predict(model, np.zeros((50, 50), dtype = np.uint8))		

#print(x)

recognize()
