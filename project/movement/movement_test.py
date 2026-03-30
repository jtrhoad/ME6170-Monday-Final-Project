
#!/usr/bin/env python3

import sys
import time  # Explicit import instead of relying on wildcard

sys.path.append('/home/pi/project_demo/lib')
from McLumk_Wheel_Sports import *

def main(speed=50, duration=1):
    """
    Moves the Mecanum robot forward then backward.
    
    Args:
        speed (int): Motor speed (0-100)
        duration (float): Duration in seconds for each movement
    """
    try:
        print("Moving forward...")
        move_forward(speed)
        time.sleep(duration)
        stop_robot()
        time.sleep(0.5)

        print("Moving backward...")
        move_backward(speed)
        time.sleep(duration)
        stop_robot()
        time.sleep(0.5)

        print("Done.")

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    finally:
        # Always stop the robot, whether finished or interrupted
        stop_robot()
        print("Robot stopped.")


if __name__ == "__main__":
    main()