from gpiozero import Servo
from time import sleep

# Initialize servos (GPIO 17 and 22)
servo1 = Servo(17)
servo2 = Servo(22)

def set_angle(servo, angle):
    # Convert 0–180 → -1 to 1 (gpiozero range)
    value = (angle / 90) - 1
    servo.value = value

# Initial positions
set_angle(servo1, 0)
set_angle(servo2, 0)

print("Starting in 5 seconds...")
sleep(5)

while True:
    print("Running sequence...")

    # Step 1
    set_angle(servo1, 70)
    set_angle(servo2, 0)
    sleep(2)

    # Step 2
    set_angle(servo1, 45)
    sleep(1)

    # Step 3
    set_angle(servo2, 90)
    sleep(3)

    # Step 4
    set_angle(servo2, 0)
    sleep(1)

    # Step 5
    set_angle(servo1, 70)

    print("Sequence complete. Restarting...\n")