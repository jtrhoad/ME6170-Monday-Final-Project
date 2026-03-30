#!/usr/bin/env python3

import sys
import time  # Explicit import instead of relying on wildcard

sys.path.append('/home/pi/project_demo/lib')
from McLumk_Wheel_Sports import *

def main(speed=25, pulse_length=1, op_length=10):
    """
    Moves the Mecanum robot forward then backward.
    
    Args:
        speed (int): Motor speed (0-100)
        pulse_length (float): Duration in seconds for each movement
        op_length (float): Duration in seconds for
    """
    try:
        print("Beggining opperation")
        end_time = time.time() + op_length
        while end_time >time.time():
            if ((digitalread(R_s) == 0) and (digitalread(L_s) == 0)) :
                move_forward(speed)
                time.sleep(pulse_length)
                stop_robot()
            elif ((digitalread(R_s) == 1) and (digitalread(L_s) == 0)) :
                rotate_right(speed)
                time.sleep(pulse_length)
                stop_robot
            elif ((digitalread(R_s) == 0) and (digitalread(L_s) == 1)) :
                rotate_left(speed)
                time.sleep(pulse_length)
                stop_robot

        
        print("Done.")

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    finally:
        # Always stop the robot, whether finished or interrupted
        stop_robot()
        print("Robot stopped.")


if __name__ == "__main__":
    main()