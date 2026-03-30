## Script just to see if servo motor works when connected to raspbot

import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM)

SERVO_PIN = 17
GPIO.setup(SERVO_PIN, GPIO.OUT)

servo = GPIO.PWM(SERVO_PIN, 50)
servo.start(0)

def set_angle(angle):
    duty = 2 + (angle / 18)
    servo.ChangeDutyCycle(duty)
    time.sleep(0.5)
    servo.ChangeDutyCycle(0)

try:
    while True:
        print("0 degrees")
        set_angle(0)

        print("90 degrees")
        set_angle(90)

        print("180 degrees")
        set_angle(180)

except KeyboardInterrupt:
    servo.stop()
    GPIO.cleanup()