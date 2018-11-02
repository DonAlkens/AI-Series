import numpy as np
import cv2
import pickle

# Initializing the cascade classifier
face_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_alt2.xml")
eye_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_eye.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

labels = {}
with open("labels.pickle", "rb") as f:
    og_labels = pickle.load(f)
    labels = {v: k for k, v in og_labels.items()}

cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Converting to gray scale
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detecting multiple faces
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.5, minNeighbors=5)
    # Iterating over the detected faces
    for (x, y, w, h) in faces:
        # print(x, y, w, h)

        # Setting detection frame box values
        color_img = frame[y:y+h, x:x+w]
        points = gray_img[y:y+h, x:x+w]

        # recognition of the faces ? Deep learned model predict
        # Keras tensorflow pytorch scikit learn

        # saving the image
        id_, conf = recognizer.predict(points)

        if conf >= 45:
            # print(id_)

            name, ext = labels[id_].split(".")
            name = name.replace("_", " ")

            font = cv2.FONT_HERSHEY_SIMPLEX
            color = (255, 255, 255)
            stroke = 2
            cv2.putText(frame, name, (x, y), font, 1, color, stroke, cv2.LINE_AA)
            # print(name)

        image = "images/face.png"
        cv2.imwrite(image, color_img)

        # drawing a rectangle round the face
        color = (255, 0, 0)
        stroke = 2

        cord_width = x + w
        cord_height = y + h
        cv2.rectangle(frame, (x, y), (cord_width, cord_height), color, stroke)

        # Getting the eye cascade
        eyes = eye_cascade.detectMultiScale(points)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(color_img, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

    # Displaying the resulting frame
    cv2.imshow("Face Detection", frame)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# when everything is done release the window
cap.release()
cv2.destroyAllWindows()