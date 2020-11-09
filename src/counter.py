import argparse
import cv2
import imutils
import numpy as np
import os
import picamera
import time
import RPi.GPIO as GPIO
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

# initialize arg parser
parser = argparse.ArgumentParser(description='WatCounts COVID Occupancy Tracker')
parser.add_argument('-above', action='store_true', help='Camera positioned above entrance')
args = parser.parse_args()

# initialize GPIO
GPIO.setwarnings(False)		# disable warnings
GPIO.setmode(GPIO.BCM)		# program GPIO with BCM pin numbers (ie PIN32 as 'GPIO12')

# initialize DC motor GPIO control (ENA, IN1 & IN2)
# remove the jumper on ENA and connect to PWM for speed control, otherwise jumper will run DC motor at max speed
dir_pin0 = 16
dir_pin1 = 20
pwm_pin = 12

GPIO.setup(dir_pin0, GPIO.OUT)	# set dir0 pin as output (IN1)
GPIO.setup(dir_pin1, GPIO.OUT)	# set dir1 pin as output (IN2)
GPIO.setup(pwm_pin, GPIO.OUT)		# set pwm pin as output (ENA)
pwm = GPIO.PWM(pwm_pin, 1000)	# PWM output at 1kHz frequency
pwm.start(0)					# start PWM output

# create new directory to store frames
save_dir = '/home/pi/Projects/Videos/{}/'.format(datetime.now())
os.mkdir(save_dir)

# instantiate centroid tracker
ct = CentroidTracker(maxDisappeared=50, maxDistance=150)
trackableObjects = {}

W = None # width of frame
H = None # height of frame

numInLibrary = 0 # global count
prevNumInLibrary = 0 # prev global count

numExited = 0 # number of people that have exited library
numEntered = 0 # number of people that have entered library

# initialize HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
	image = frame.array
	image = imutils.resize(image, width=min(400, image.shape[1]))
	orig = image.copy()

	if H is None or W is None:
		(H, W) = image.shape[:2]

	# reset local count
	numEntered = 0
	numExited = 0

	# detect people in the image
	(rects, weights) = hog.detectMultiScale(image, winStride=(8, 8),
		padding=(16, 16), scale=1.05, useMeanshiftGrouping=False)

	# draw initial bounding boxes
	for (x, y, w, h) in rects:
		cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)

	# apply non-maxima suppression to bounding boxes (combines overlapping boxes into one)
	rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
	pick = non_max_suppression(rects, probs=None, overlapThresh=0.3)

	# draw final bounding boxes
	for (xA, yA, xB, yB) in pick:
		cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)

	# draw line across center of frame to determine crossing direction
	# start point, end point, colour, thickness
	if args.above:
		cv2.line(image, (0, H // 2), (W, H // 2), (0, 255, 255), 2) # horizontal line
	else:
		cv2.line(image, (W // 2, 0), (W // 2, H), (0, 255, 255), 2) # vertical line

	# use centroid tracker to associate old centroids with new centroids
	objects = ct.update(rects)

	# loop over tracked objects
	for (objectID, centroid) in objects.items():
		to = trackableObjects.get(objectID, None)

		# if there is no existing trackable object, create one
		if to is None:
			to = TrackableObject(objectID, centroid)

		else:
			# difference between coord of current centroid and mean of previous centroids gives direction 

			# (down positive, up negative)
			if args.above:
				y = [c[1] for c in to.centroids]
				direction = centroid[1] - np.mean(y)

			# (left positive, right negative)
			else:
				x = [c[0] for c in to.centroids]
				direction = centroid[0] - np.mean(x)

			to.centroids.append(centroid)

			# check if object has been counted or not
			if not to.counted:
				if args.above:
					# if direction is negative (object is moving up) and centroid is above center line, count object
					if direction < 0 and centroid[1] < H // 2:
						numEntered += 1
						to.counted = True

					# if direction is positive (object is moving down) and centroid is below center line, count object
					elif direction > 0 and centroid[1] > H // 2:
						numExited += 1
						to.counted = True

				else:
					# if direction is negative (object is moving left) and centroid is left of center line, count object
					if direction < 0 and centroid[0] < W // 2:
						numEntered += 1
						to.counted = True

					# if direction is positive (object is moving right) and centroid is right of center line, count object
					elif direction > 0 and centroid[0] > W // 2:
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

	# print message if count was updated
	if (numInLibrary != prevNumInLibrary):
		print("Num in library: {}, num exited: {}, num entered: {}".format(numInLibrary, numExited, numEntered))

	# activate locking system if needed
	"""	Directional Output to L298 motor controller
	IN1	IN2
	0	0	stop
	0	1	clockwise
	1	0	counter-clockwise
	1	1	stop
	"""
	if (prevNumInLibrary <= 45 and numInLibrary > 45):
		print("Activating lock")
		GPIO.output(dir_pin0, 0)
		GPIO.output(dir_pin1, 1)
		pwm.ChangeDutyCycle(50)
	
	elif (prevNumInLibrary > 45 and numInLibrary < 45):
 		print("Deactivating lock")
		GPIO.output(dir_pin0, 1)
		GPIO.output(dir_pin1, 0)
		pwm.ChangeDutyCycle(50)
	
	time.sleep(2)

	# stop motor
	GPIO.output(dir_pin0, 0)
	GPIO.output(dir_pin1, 0)
	pwm.ChangeDutyCycle(0)

	prevNumInLibrary = numInLibrary

	# clear stream for next frame
	rawCapture.truncate(0)