#!/usr/bin/env python3

import sys
import time
import math
from Raspbot_Lib import Raspbot

bot = Raspbot()

def main():
    bot.Ctrl_Ulatist_Switch(1)

    try:
        dist_H = bot.read_data_array(0x1b, 1)[0]
        dist_L = bot.read_data_array(0x1a, 1)[0]
        dist   = (dist_H << 8 | dist_L) / 10.0

        terminal_print(f"Distance: {dist:.1f} cm")

        if dist < 15:
            bot.Ctrl_WQ2812_ALL(1, 0)   # red   — very close
        elif dist < 30:
            bot.Ctrl_WQ2812_ALL(1, 3)   # yellow — medium
        else:
            bot.Ctrl_WQ2812_ALL(1, 1)   # green  — clear

        time.sleep(0.5)

    finally:
        bot.Ctrl_Ulatist_Switch(0)
        bot.Ctrl_WQ2812_ALL(0, 0)
        del bot
        terminal_print("Sensor off, lights off, bot reset. Done.")

if __name__ == "__main__":
    main()