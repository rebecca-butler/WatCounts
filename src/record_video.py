import picamera
from datetime import date
from time import sleep

camera = picamera.PiCamera()
camera.start_preview()
today = date.today()
camera.start_recording('/home/pi/Projects/Videos/{}.h264'.format(today))
sleep(5)
camera.stop_recording()
camera.stop_preview()