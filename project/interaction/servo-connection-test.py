#!/usr/bin/env python3
"""
servo_all_test.py
=================
Tests all 4 servos together:
  - Camera pan/tilt (servos 1 & 2) via expansion board (I2C 0x2B)
  - Extra servos via GPIO 18 & 19 (hardware PWM, lgpio backend)
"""
from gpiozero import AngularServo
from gpiozero.pins.lgpio import LGPIOFactory
from Raspbot_Lib import Raspbot
from time import sleep

# --- Camera servos via expansion board ---
bot = Raspbot()
PAN_ID  = 1
TILT_ID = 2

# --- Extra servos via GPIO ---
factory = LGPIOFactory()
servo3 = AngularServo(18, min_angle=0, max_angle=180,
                      min_pulse_width=0.0005, max_pulse_width=0.0025,
                      pin_factory=factory)
servo4 = AngularServo(19, min_angle=0, max_angle=180,
                      min_pulse_width=0.0005, max_pulse_width=0.0025,
                      pin_factory=factory)

def move_all(pan_angle, tilt_angle, gpio_angle, label):
    print(f"\n--- {label} ---")
    print(f"  Camera pan={pan_angle}  tilt={tilt_angle}")
    print(f"  GPIO  servo3={gpio_angle}  servo4={gpio_angle}")
    bot.Ctrl_Servo(PAN_ID,  pan_angle)
    bot.Ctrl_Servo(TILT_ID, tilt_angle)
    servo3.angle = gpio_angle
    servo4.angle = gpio_angle
    sleep(1.5)

try:
    while True:
        move_all(pan_angle=72,  tilt_angle=25,  gpio_angle=0,   label="Start position")
        move_all(pan_angle=0,   tilt_angle=5,   gpio_angle=90,  label="Min pan/tilt, mid GPIO")
        move_all(pan_angle=175, tilt_angle=95,  gpio_angle=180, label="Max pan/tilt, max GPIO")
        move_all(pan_angle=72,  tilt_angle=25,  gpio_angle=90,  label="Center all")

except KeyboardInterrupt:
    # Return camera to center before exiting
    bot.Ctrl_Servo(PAN_ID,  72)
    bot.Ctrl_Servo(TILT_ID, 25)
    servo3.detach()
    servo4.detach()
    print("\nAll servos stopped.")