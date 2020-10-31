import picamera
import cv2
import time
import datetime
from picamera.array import PiRGBArray

# initialize camera settings
camera = picamera.PiCamera()
camera.framerate = 32
camera.resolution = (640, 480)
rawCapture = PiRGBArray(camera, size=(640, 480))
time.sleep(0.1)
now = date.now()

# capture frames from camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # show frame
	image = frame.array
	cv2.imshow("Frame", image)
	key = cv2.waitKey(1) & 0xFF

    # save frame on disk
    cv2.imwrite('/home/pi/Projects/Videos/{}.jpg'.format(datetime.datetime.now()), image)

    # clear stream for next frame
	rawCapture.truncate(0)

	# break if q is pressed
	if key == ord("q"):
		break