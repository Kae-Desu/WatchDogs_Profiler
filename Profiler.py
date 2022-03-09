# Project made by me かえです, inspired by Watchdogs franchise.
# This is the profiler (that cool stuff)
# How this work: This program use camera to scan, recognize and detect possible IoT product, Smartphone, People, Camera, and much more
# For now only face regocnition (pardon my typo and bad english)

import sys
import cv2
import numpy as np
import pickle
import itertools
import threading
import time
from turtle import color
from colorama import init
init(strip = not sys.stdout.isatty())
from termcolor import cprint
from pyfiglet import figlet_format

# ART
cprint(figlet_format('Profiler', font = 'doom'), 'white', attrs=['bold'])
print("By: かえです")
print("Ver.: Beta 0.0.1")
print("Starting Camera")

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
'''
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")
'''
'''
labels = {}
with open("labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k, v in og_labels.items()}
'''
    
vid = cv2.VideoCapture(0)

# defining resolution
def res_1080p():
    vid.set(3,1920)
    vid.set(4,1080)

# res_1080p() #change Windows size & camera res to 1080p

while True:
    # Capture frame-by-frame
    ret, frame = vid.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor = 1.5,
        minNeighbors = 5,
        minSize = (30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, h, w) in faces:
        cv2.rectangle(frame, (x, y), (x+h, y+w), (0, 255, 0), 2)
        cv2.rectangle(frame, (x+h, y+w), (x+h+h, y-w-y), (255, 0, 0), cv2.FILLED)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Recognizer
        '''
        id_, conf = recognizer.predict(roi_gray)
        if conf >= 45 and conf <= 85:
            name = labels[id_]
            cv2.putText(frame, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
        '''
        # Print your image out
        img_item = "my-image.png"
        cv2.imwrite(img_item, roi_gray)

    # Display the resulting frame
    cv2.imshow('Webcam', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
vid.release()
cv2.destroyAllWindows()




# Changelog
'''
Version    |    Date & Time         |  Changes
===========|========================|==================
Beta 0.0.1 |    17/01/2022          |  Project Started
no change  |    03/09/2022          |  Seems like there is an error because there isn't any 'trainner.yml' file, tried to make an if function for trainner.yml

I'm starting this project ^_^ "yeah"

'''

''' (this is some kind of notes hahah, pretty convenient i might say)
testing if else func

from os.path import exists as ada
print("yes") if ada('FaceTrain') else print("nope")
'''