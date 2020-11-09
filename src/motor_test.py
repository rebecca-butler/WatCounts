import time
import RPi.GPIO as GPIO

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
pwm.start(0)	

# activate locking system
print("Activating lock")
GPIO.output(dir_pin0, 0)
GPIO.output(dir_pin1, 1)
pwm.ChangeDutyCycle(50)
	
time.sleep(2)

# stop motor
GPIO.output(dir_pin0, 0)
GPIO.output(dir_pin1, 0)
pwm.ChangeDutyCycle(0)

time.sleep(3)

# deactivate locking system
print("Deactivating lock")
GPIO.output(dir_pin0, 1)
GPIO.output(dir_pin1, 0)
pwm.ChangeDutyCycle(50)
	
time.sleep(2)

# stop motor
GPIO.output(dir_pin0, 0)
GPIO.output(dir_pin1, 0)
pwm.ChangeDutyCycle(0)