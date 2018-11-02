import cv2
import numpy as np
import os
from PIL import Image
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "images")

# Initializing the cascade classifier
face_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_alt2.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()


# Creating the Model Label
current_id = 0
label_id = {}
y_labels = []
x_train = []

for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root, file)
            label = os.path.basename(file).replace(" ", "-").lower()

            # print(label, path)

            if not label in label_id:
                label_id[label] = current_id
                current_id += 1

            id_ = label_id[label]
            # print(label_id)

            # y_labels.append(label)
            # x_train.append(path)

            pil_image = Image.open(path).convert("L")  # converting gray scale
            size = (550, 550)
            final_image = pil_image.resize(size, Image.ANTIALIAS) # resizing the images
            image_arr = np.array(pil_image, "uint8")

            # print(image_arr)

            # detecting multiple faces
            faces = face_cascade.detectMultiScale(image_arr, scaleFactor=1.5, minNeighbors=5)

            # Training the model with image arr
            for (x, y, w, h) in faces:
                point = image_arr[y:y+h, x:x+w]
                x_train.append(point)
                y_labels.append(id_)


# print(y_labels)
# print(x_train)

# saving the labels with pickle
with open("labels.pickle", "wb") as f:
    pickle.dump(label_id, f)

# training the Item for recognition
recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainer.yml")