import imutils
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import os
import time
from datetime import date, datetime
from centroidtracker import CentroidTracker
from trackableobject import TrackableObject
import picamera
from picamera.array import PiRGBArray

# initialize camera settings
camera = picamera.PiCamera()
camera.framerate = 32
camera.resolution = (640, 480)
rawCapture = PiRGBArray(camera, size=(640, 480))
time.sleep(0.1)
now = date.now()

ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
trackableObjects = {}

W = None # width of frame
H = None # height of frame

numInLibrary = 0 # global count
numExited = 0 # number of people that have exited library (moved down)
numEntered = 0 # number of people that have entered library (moved up)

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    image = frame.array
    frame = imutils.resize(frame, width=500)
    orig = image.copy()

	if H is None or W is None:
    	(H, W) = frame.shape[:2]

    # detect people in the image
    (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),
        padding=(8, 8), scale=1.05)

    # draw initial bounding boxes
    for (x, y, w, h) in rects:
        cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # apply non-maxima suppression to bounding boxes (combines overlapping boxes into one)
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

    # draw final bounding boxes
    for (xA, yA, xB, yB) in pick:
        cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)

    # draw horizontal line across frame to determine crossing direction
    cv2.line(frame, (0, H // 2), (W, H // 2), (0, 255, 255), 2)

    # use centroid tracker to associate old centroids with new centroids
    objects = ct.update(rects)

    # loop over tracked objects
    for (objectID, centroid) in objects.items():
        to = trackableObjects.get(objectID, None)

        # if there is no existing trackable object, create one
        if to is None:
            to = TrackableObject(objectID, centroid)

        else:
            # difference between y-coord of current centroid and mean of previous 
            # centroids gives direction (down positive, up negative)
            y = [c[1] for c in to.centroids]
            direction = centroid[1] - np.mean(y)
            to.centroids.append(centroid)

            # check if object has been counted or not
            if not to.counted:
                # if direction is negative (object is moving up) and centroid is above center line, count object
                if direction < 0 and centroid[1] < H // 2:
                    numEntered += 1
                    to.counted = True

                # if direction is positive (object is moving down) and centroid is below center line, count object
                elif direction > 0 and centroid[1] > H // 2:
                    numExited += 1
                    to.counted = True

        # store trackable object in dict
        trackableObjects[objectID] = to

        # draw ID of object and centroid of object on output frame
        text = "ID {}".format(objectID)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

    # show and save output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    cv2.imwrite('/home/pi/Projects/Videos/{}.jpg'.format(datetime.datetime.now()), frame)

	# break if q pressed
    if key == ord("q"):
        break

    # update global count
    numInLibrary = numInLibrary + numEntered - numExited
    print("Num in library: {}, num exited: {}, num entered: {}".format(numInLibrary, numExited, numEntered))

    # activate locking system if needed
    if (numInLibrary > 45):
        print("Maximum reached")
        # activate lock
    else:
        # deactivate lock
        print("Maximum has not been reached")