import RPi.GPIO as GPIO
import time

# Use BCM pin numbering
GPIO.setmode(GPIO.BCM)

# Servo pins
servo1_pin = 17
servo2_pin = 22

# Setup pins
GPIO.setup(servo1_pin, GPIO.OUT)
GPIO.setup(servo2_pin, GPIO.OUT)

# Setup PWM (50 Hz for servos)
servo1 = GPIO.PWM(servo1_pin, 50)
servo2 = GPIO.PWM(servo2_pin, 50)

servo1.start(0)
servo2.start(0)

def set_angle(pwm, angle):
    # Convert angle (0–180) to duty cycle (approx 2.5–12.5)
    duty = 2.5 + (angle / 180.0) * 10
    pwm.ChangeDutyCycle(duty)
    time.sleep(0.3)  # allow servo to move
    pwm.ChangeDutyCycle(0)  # stop jitter

try:
    # Initialize positions
    set_angle(servo1, 0)
    set_angle(servo2, 0)

    print("Starting in 5 seconds...")
    time.sleep(5)

    while True:
        print("Running sequence...")

        # Step 1
        set_angle(servo1, 70)
        set_angle(servo2, 0)
        time.sleep(2)

        # Step 2
        set_angle(servo1, 45)
        time.sleep(1)

        # Step 3
        set_angle(servo2, 90)
        time.sleep(3)

        # Step 4
        set_angle(servo2, 0)
        time.sleep(1)

        # Step 5
        set_angle(servo1, 70)

        print("Sequence complete. Restarting...\n")

except KeyboardInterrupt:
    print("Stopping...")

finally:
    servo1.stop()
    servo2.stop()
    GPIO.cleanup()