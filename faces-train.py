import cv2
import os
import numpy as np
from PIL import Image
import pickle
import splitfolders 

def norm(img):
	norm_img = np.zeros((300, 300))
	norm_img = cv2.normalize(img, norm_img, 0, 255, cv2.NORM_MINMAX)
	
	return norm_img

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "images")

splitfolders.ratio(image_dir,'C:/Users/naila/OneDrive/Bureau/test/output/',seed=1337,ratio=(0.8,0.2))

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id = 0
label_ids = {}
y_labels = []
x_train = []

for root, dirs, files in os.walk(image_dir):
	for file in files:
		if file.endswith("png") or file.endswith("jpg"):
			path = os.path.join(root, file)
			label = os.path.basename(root).replace(" ", "-").lower()
			
			if not label in label_ids:
				label_ids[label] = current_id
				current_id += 1
			id_ = label_ids[label]
			
			pil_image = Image.open(path).convert("L") # grayscale
			image_array = np.array(pil_image, "uint8")
			image_array = norm(image_array)
			faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)

			for (x,y,w,h) in faces:
				roi = image_array[y:y+h, x:x+w]
				x_train.append(roi)
				y_labels.append(id_)

#save label id
with open("./pickles/face-labels.pickle", 'wb') as f:
	pickle.dump(label_ids, f)

recognizer.train(x_train, np.array(y_labels))
recognizer.save("./recognizers/face-trainner.yml")