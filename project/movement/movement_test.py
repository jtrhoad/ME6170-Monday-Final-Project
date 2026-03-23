#!/usr/bin/env python3

import sys
sys.path.append('/home/pi/project_demo/lib')
#导入麦克纳姆小车驱动库 Import Mecanum Car Driver Library
from McLumk_Wheel_Sports import *

def main():
    # 定义移动速度和持续时间 Define movement speed and duration
    duration = 1
    speed = 100
    try:
        move_right(speed)
        time.sleep(duration)
        stop_robot()
        time.sleep(1)
        move_left(speed)
        time.sleep(duration)
        stop_robot()
        time.sleep(1)


    except KeyboardInterrupt:
        # 当用户按下停止时，停止小车运动功能 When the user presses the stop button, the car stops moving.
            stop_robot()
            print("off.")


if __name__ == "__main__":
    main()
    #使用完成对象记住释放掉对象，不然下一个程序使用这个对象模块会被占用，导致无法使用
#Release Object
del bot     