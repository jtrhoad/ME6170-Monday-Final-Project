#!/usr/bin/env python3
from gpiozero import AngularServo
from gpiozero.pins.lgpio import LGPIOFactory
from time import sleep

# Force lgpio backend (required for Pi 5)
factory = LGPIOFactory()

servo1 = AngularServo(18, min_angle=0, max_angle=180,
                      min_pulse_width=0.0005, max_pulse_width=0.0025,
                      pin_factory=factory)
servo2 = AngularServo(19, min_angle=0, max_angle=180,
                      min_pulse_width=0.0005, max_pulse_width=0.0025,
                      pin_factory=factory)

try:
    while True:
        print("0 degrees")
        servo1.angle = 0
        servo2.angle = 0
        sleep(1)

        print("90 degrees")
        servo1.angle = 90
        servo2.angle = 90
        sleep(1)

        print("180 degrees")
        servo1.angle = 180
        servo2.angle = 180
        sleep(1)

except KeyboardInterrupt:
    servo1.detach()
    servo2.detach()
    print("Done.")