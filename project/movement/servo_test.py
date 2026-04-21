#!/usr/bin/env python3
# claw_angle_finder.py
# Run this to find the correct CLAW_CLOSED and CLAW_OPEN angles for your servo

import time
from gpiozero import AngularServo
from gpiozero.pins.lgpio import LGPIOFactory

factory = LGPIOFactory()
claw = AngularServo(19, min_angle=0, max_angle=180,
                    min_pulse_width=0.0005, max_pulse_width=0.0025,
                    pin_factory=factory)

print("=== Claw Angle Finder ===")
print("Enter angles between 0 and 180 to move the claw.")
print("Find the angle where it is FULLY OPEN and FULLY CLOSED.")
print("Type 'q' to quit.\n")

current = 90
claw.angle = current
print(f"Starting at angle: {current}")
time.sleep(1)

while True:
    user_input = input("\nEnter angle (0-180) or 'q' to quit: ").strip()

    if user_input.lower() == 'q':
        break

    try:
        angle = float(user_input)
        if 0 <= angle <= 180:
            claw.angle = angle
            current = angle
            print(f"Moved to {angle} degrees")
            time.sleep(0.5)
        else:
            print("Please enter a value between 0 and 180")
    except ValueError:
        print("Invalid input — enter a number between 0 and 180")

claw.value = None
print(f"\nLast angle was: {current}")
print("Update CLAW_CLOSED and CLAW_OPEN in V10 with the angles you found.")
print("Done.")