#!/usr/bin/env python3

"""
import sys
sys.path.append(r"C:\Users\Josh\OneDrive - Clemson University\Documents\GitHub\ME6170-Monday-Final-Project\external\Libraries\raspbot")
from Raspbot_Lib import Raspbot
"""
"""
import sys
sys.path.append(r"C:\Users\Josh\OneDrive - Clemson University\Documents\GitHub\ME6170-Monday-Final-Project\external\Libraries\lib")
#导入麦克纳姆小车驱动库 Import Mecanum Car Driver Library
from McLumk_Wheel_Sports import *
"""

from ...external.Libraries.lib.McLumk_Wheel_Sports import *

def main():
    # 定义移动速度和持续时间 Define movement speed and duration
    duration = 1
    speed = 50
    try:
        move_forward(speed)
        time.sleep(duration)
        stop_robot()
        time.sleep(0.5)
        move_backward(speed)
        time.sleep(duration)
        stop_robot()
        time.sleep(0.5)
    except KeyboardInterrupt:
        # 当用户按下停止时，停止小车运动功能 When the user presses the stop button, the car stops moving.
            stop_robot()
            print("off.")


if __name__ == "__main__":
    main()