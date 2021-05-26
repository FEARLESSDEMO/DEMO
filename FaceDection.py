#COMPUTER VISION PROJECT
#PROJECT FROM FEARLESSBEE TO SHAPE AI
import cv2
import numpy as np
from trainModel import *

cascade_path = 'cascades/haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)


#FIND FACE FROM ORIGINAL FACE


def detect_face(image):
	gray = cv2.cvtColor(image, cv2.COLOR_GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)

	if faces is ():
		return image

	for (x, y, w, h) in faces:
		cv2.rectangle(image, (x,y), (x+w, y+h), (0,255,0), 2)
		crop_face = image[y:y+h, x:x+w]
		crop_face = cv2.resize(crop_face, (200,200))
	
	return crop_face		

capture = cv2.VideoCapture(0)

while True:
	response, frame = capture.read()
	found_face = detect_face(frame)	

	try:
		gray_face = cv2.cvtColor(found_face, cv2.COLOR_GRAY)

		# predict using trained model

		label, score = classifier.predict(gray_face)
		if score < 500:
			confidence_score = int(100 * (1 - score/400))
			message = 'I am {} confident '.format(str(confidence_score))

		cv2.putText(frame, message, (50,50), cv2.FONT_ITALIC, 1, (200,0,200), 2)

		if score > 75:
			message = 'SORRY YOU ARE NOT IN ZONE'
			cv2.putText(frame, message, (160,50), cv2.FONT_ITALIC, 1, (200,200,0), 2)
			cv2.imshow('I recognize You', image)
		else:
			message = 'YOU ARE IN SAFEZONE'
			cv2.putText(frame, message, (160,50), cv2.FONT_ITALIC, 1, (0,200,200), 2)	
			cv2.imshow('I recognize You', image)
	
	except:
		message = 'NO FACE FOUND!'
		cv2.putText(frame, message, (180,180), cv2.FONT_ITALIC, 1, (0,200,200), 2)	
		cv2.imshow('I recognize You', frame)
		pass

	if cv2.waitKey(0):
		break

capture.release()
#Application Will Closed

cv2.destroyAllWindows()			

				
