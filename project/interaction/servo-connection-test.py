import pigpio
import time
 
# Connect to pigpio daemon
pi = pigpio.pi()
 
if not pi.connected:
    print("Failed to connect to pigpio daemon")
    exit()
 
# Define GPIO pins
SERVO1 = 17
SERVO2 = 7
 
# Function to move servo (pulse width in microseconds)
def set_servo(pin, pulse):
    pi.set_servo_pulsewidth(pin, pulse)
 
try:
    while True:
        # Move to minimum (~0 degrees)
        set_servo(SERVO1, 500)
        set_servo(SERVO2, 500)
        time.sleep(1)
 
        # Move to middle (~90 degrees)
        set_servo(SERVO1, 1500)
        set_servo(SERVO2, 1500)
        time.sleep(1)
 
        # Move to maximum (~180 degrees)
        # set_servo(SERVO1, 2500)
        # set_servo(SERVO2, 2500)
        # time.sleep(1)
 
except KeyboardInterrupt:
    pass
 
# Stop servos
set_servo(SERVO1, 0)
set_servo(SERVO2, 0)
 
pi.stop()