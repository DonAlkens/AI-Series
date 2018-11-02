import numpy as np
import cv2
import os

filename = 'video.avi'
frames_per_second = 24.0
res = '720p'


def change_res(cp, width, height):
    cp.set(3, width)
    cp.set(4, height)


# Standard Video Dimensions Sizes
STD_DIMENSIONS = {
    "480p": (640, 480),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "4k": (3840, 2160),
}


# grab resolution dimensions and set video capture to it.
def get_dims(cp, res='1080p'):
    width, height = STD_DIMENSIONS["480p"]
    if res in STD_DIMENSIONS:
        width, height = STD_DIMENSIONS[res]
    # change the current caputre device
    # to the resulting resolution
    change_res(cp, width, height)
    return width, height


# Video Encoding, might require additional installs
# Types of Codes: http://www.fourcc.org/codecs.php
VIDEO_TYPE = {
    'avi': cv2.VideoWriter_fourcc(*'XVID'),
    # 'mp4': cv2.VideoWriter_fourcc(*'H264'),
    'mp4': cv2.VideoWriter_fourcc(*'XVID'),
}


def get_video_type(filename):
    filename, ext = os.path.splitext(filename)
    if ext in VIDEO_TYPE:
        return VIDEO_TYPE[ext]
    return VIDEO_TYPE['avi']


def faceDetection(test_img):
    pass


cap = cv2.VideoCapture(0)
out = cv2.VideoWriter(filename, get_video_type(filename), 25, get_dims(cap, res))

# width = 500
# height = 500
# change_res(cap, width, height)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    out.write(frame)
    cv2.imshow('Camera', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# when everything is done
cap.release()
out.release()
cv2.destroyAllWindows()