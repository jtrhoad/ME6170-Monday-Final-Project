#!/usr/bin/env python3

import sys
import time
import tty
import termios
import threading

sys.path.append('/home/pi/project_demo/lib')
from Raspbot_Lib import Raspbot

def terminal_print(msg):
    sys.stdout.write(msg + "\r\n")
    sys.stdout.flush()

bot = Raspbot()
quit_flag = threading.Event()

def watch_for_q():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        while not quit_flag.is_set():
            ch = sys.stdin.read(1)
            if ch.lower() == 'q':
                terminal_print("\r\nQ pressed — shutting down...")
                quit_flag.set()
                break
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

def main():
    key_thread = threading.Thread(target=watch_for_q, daemon=True)
    key_thread.start()

    terminal_print("=== Ultrasonic Sensor Test ===")
    terminal_print("Press Q or Ctrl+C to stop.")
    terminal_print("")

    bot.Ctrl_Ulatist_Switch(1)

    try:
        while not quit_flag.is_set():
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

    except KeyboardInterrupt:
        terminal_print("\r\nCtrl+C pressed — shutting down...")
        quit_flag.set()

    finally:
        bot.Ctrl_Ulatist_Switch(0)
        bot.Ctrl_WQ2812_ALL(0, 0)
        del bot
        terminal_print("Sensor off, lights off, bot reset. Done.")

if __name__ == "__main__":
    main()