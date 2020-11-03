import picamera
import cv2
import time
import os
from datetime import date, datetime
from picamera.array import PiRGBArray

# initialize camera settings
camera = picamera.PiCamera()
camera.framerate = 32
camera.resolution = (640, 480)
rawCapture = PiRGBArray(camera, size=(640, 480))
time.sleep(0.1)

# create new directory to store frames
save_dir = '/home/pi/Projects/Videos/{}/'.format(date.today())
os.mkdir(save_dir)

# capture frames from camerap
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # show frame
    image = frame.array
    cv2.imshow("Frame", image)
    key = cv2.waitKey(1) & 0xFF

    # store frame on disk
    file_path = save_dir + '{}.jpg'.format(datetime.now())
    cv2.imwrite(file_path, image)

    # clear stream for next frame
    rawCapture.truncate(0)
    
    # break if q is pressed
    if key == ord("q"):
        break