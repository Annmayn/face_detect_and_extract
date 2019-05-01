import numpy as np 
import cv2
from os import listdir, mkdir
from os.path import join, isfile

root_path = "E:/Machine Learning/Datasets/"
dataset_name = "orl_faces/"
dataset_path = root_path+dataset_name
face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def face_extractor(img_path):
	img = cv2.imread(img_path)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = face_classifier.detectMultiScale(gray,1.3,5)
	if faces is():
		return None
	for(x,y,w,h) in faces:
		cropped_faces = img[y:y+h, x:x+w]
		return cropped_faces

for img_name in listdir(dataset_path):
	img_dir = dataset_path + img_name + '/'
	only_images = [f for f in listdir(img_dir) if isfile(join(img_dir,f))]

	for img in only_images:
		img_path = img_dir+img
		print(img_path)
		img = cv2.imread(img_path)
		cropped_img = face_extractor(img_path) 
		if cropped_img is not None:
			cv2.imshow('test', img)
			cv2.waitKey(0)	#Esc to close

			cv2.imshow('test', cropped_img)
			cv2.waitKey(0)	#Esc to close
			cv2.destroyAllWindows()

	#save the modifed file