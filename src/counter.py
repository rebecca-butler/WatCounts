import cv2
import imutils
import numpy as np
import os
import picamera
import time
from datetime import datetime
from imutils.object_detection import non_max_suppression
from picamera.array import PiRGBArray

# custom classes
from centroidtracker import CentroidTracker
from trackableobject import TrackableObject

# initialize camera settings
camera = picamera.PiCamera()
camera.framerate = 32
camera.resolution = (640, 480)
rawCapture = PiRGBArray(camera, size=(640, 480))
time.sleep(0.1)

# create new directory to store frames
save_dir = '/home/pi/Projects/Videos/{}/'.format(datetime.now())
os.mkdir(save_dir)

# instantiate centroid tracker
ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
trackableObjects = {}

W = None # width of frame
H = None # height of frame

numInLibrary = 0 # global count
numExited = 0 # number of people that have exited library (moved down)
numEntered = 0 # number of people that have entered library (moved up)

# initialize HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
	image = frame.array
	orig = image.copy()

	if H is None or W is None:
		(H, W) = image.shape[:2]

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
	# start point, end point, colour, thickness
	cv2.line(image, (0, H // 2), (W, H // 2), (0, 255, 255), 2)

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

		# draw each object's ID and centroid on output frame
		text = "ID {}".format(objectID)
		cv2.putText(image, text, (centroid[0] - 10, centroid[1] - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
		cv2.circle(image, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

	# show and save output frame
	cv2.imshow("Frame", image)
	key = cv2.waitKey(1) & 0xFF
	cv2.imwrite(save_dir + str(datetime.now()) + '.jpg', image)

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

	# clear stream for next frame
	rawCapture.truncate(0)