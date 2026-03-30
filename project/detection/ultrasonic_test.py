
#!/usr/bin/env python3

import sys
import time

sys.path.append('/home/pi/project_demo/lib')
from Raspbot_Lib import Raspbot

bot = Raspbot()

def main():
    bot.Ctrl_Ulatist_Switch(1) # Enable ultrasonic sensor

    while True:
        time.sleep(0.5) # sensor delay
        key = get_single_keypress()
        # Read distance from sensor
        dist_H =bot.read_data_array(0x1b,1)[0]
        dist_L =bot.read_data_array(0x1a,1)[0]
        dist = (dist_H << 8 | dist_L)/10.0 # Convert to cm
        print(f"Distance: {dist} cm")
        if dist < 15: # If an obstacle is detected within 15 cm
            bot.Ctrl_WQ2812_ALL(1,0)
        elif dist < 30: # If an obstacle is detected within 30 cm
            bot.Ctrl_WQ2812_ALL(1,3)
        else: # If an obstacle is not detected within 30 cm
            bot.Ctrl_WQ2812_ALL(1,1)
        if key == 'q': # Exit on 'q' key press
            bot.Ctrl_Ulatist_Switch(0) # disable ultrasonic sensor
            bot.Ctrl_WQ2812_ALL(0,0) # turn off light
            break


        


if __name__ == "__main__":
    main()

