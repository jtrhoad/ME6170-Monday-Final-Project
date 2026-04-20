#!/usr/bin/env python3
"""
Moves arm to grab block
"""

from gpiozero import AngularServo
from gpiozero.pins.lgpio import LGPIOFactory
from Raspbot_Lib import Raspbot
from time import sleep

bot = Raspbot()

# --- Arm servos via GPIO ---
factory = LGPIOFactory()
servo3 = AngularServo(18, min_angle=0, max_angle=180,
                      min_pulse_width=0.0005, max_pulse_width=0.0025,
                      pin_factory=factory)
servo4 = AngularServo(19, min_angle=0, max_angle=180,
                      min_pulse_width=0.0005, max_pulse_width=0.0025,
                      pin_factory=factory)

# Initializing Variables
i = 1
j = 1

try:
    if i == 1:
        print("Grabbing Block Sequence")
        # Spin bot
        servo3.angle = 70
        servo4.angle = 0
        sleep(1.5)
        # Back bot into block
        servo3.angle = 0
        sleep(1.5)
        servo4.angle = 70
        i = 0

    if j == 1:
        print("Placing Block Sequence")
        # Spin bot
        servo4.angle = 0
        sleep(1.5)
        # Back bot to place
        servo3.angle = 70
        sleep(1.5)
        # Drive bot forward
        servo4.angle = 70
        j = 0

except KeyboardInterrupt:
    # Stop servos
    servo3.close()
    servo4.close()
    print("\nAll servos stopped.")