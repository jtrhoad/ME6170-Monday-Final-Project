#!/usr/bin/env python3

import sys
import time
import threading
import tty
import termios
import cv2
import numpy as np

sys.path.append('/home/pi/project_demo/lib')
from McLumk_Wheel_Sports import *
import PID

def terminal_print(msg):
    sys.stdout.write(msg + "\r\n")
    sys.stdout.flush()

# ── Camera setup ──────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
cap.set(5, 30)

# ── Servo reset (center camera) ───────────────────────────────────────────────
bot.Ctrl_Servo(1, 90)
bot.Ctrl_Servo(2, 0)

# ── PID controller ────────────────────────────────────────────────────────────
Z_axis_pid = PID.PositionalPID(0.5, 0, 1)

# ── Tracking state ────────────────────────────────────────────────────────────
prev_left  = 0
prev_right = 0

# ── Perspective transform points ──────────────────────────────────────────────
MAT_SRC = np.float32([[0, 149], [320, 149], [281, 72], [43, 72]])
MAT_DST = np.float32([[0, 240], [320, 240], [320, 0],  [0, 0]])

SPEED = 15

quit_flag = threading.Event()

# ── Q key listener ────────────────────────────────────────────────────────────
def watch_for_q():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        while not quit_flag.is_set():
            ch = sys.stdin.read(1)
            if ch.lower() == 'q':
                terminal_print("\r\nQ pressed — stopping...")
                quit_flag.set()
                break
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

# ── Line follow loop ──────────────────────────────────────────────────────────
def line_follow():
    global prev_left, prev_right

    bot.Ctrl_Ulatist_Switch(1)

    while not quit_flag.is_set():
        ret, frame = cap.read()
        if not ret:
            terminal_print("Camera read failed.")
            break

        frame       = cv2.resize(frame, (320, 240))
        M           = cv2.getPerspectiveTransform(MAT_SRC, MAT_DST)
        dst         = cv2.warpPerspective(frame, M, (320, 240))
        gray        = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
        _, bw       = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
        bw          = cv2.erode(bw, None, iterations=1)

        histogram   = np.sum(bw[bw.shape[0] // 2:, :], axis=0)
        leftx_base  = np.argmin(histogram[:320])
        rightx_base = 319 - np.argmin(histogram[::-1][:320])
        lane_center = int((leftx_base + rightx_base) / 2)
        Bias        = 159 - lane_center

        Z_axis_pid.SystemOutput = Bias
        Z_axis_pid.SetStepSignal(20)
        Z_axis_pid.SetInertiaTime(0.5, 0.2)
        pid_out = max(-20, min(20, Z_axis_pid.SystemOutput))

        diss_H      = bot.read_data_array(0x1b, 1)[0]
        diss_L      = bot.read_data_array(0x1a, 1)[0]
        distance_mm = (diss_H << 8) | diss_L

        if distance_mm < 200:
            stop_robot()
            bot.Ctrl_BEEP_Switch(1)
            time.sleep(0.05)
            bot.Ctrl_BEEP_Switch(0)

        else:
            if leftx_base == 0 and rightx_base == 319:
                if prev_left > prev_right:
                    rotate_left(SPEED)
                elif prev_right > prev_left:
                    rotate_right(SPEED)
                prev_left  = 0
                prev_right = 0

            else:
                if Bias > 35:
                    if Bias > 140:
                        rotate_left(max(1, int((SPEED - pid_out) / 5)))
                        prev_left = prev_right = 0
                    elif Bias < 50:
                        move_left(max(1, int(SPEED / 15)))
                    else:
                        move_forward(SPEED)
                    time.sleep(0.001)

                elif Bias < -35:
                    if Bias < -140:
                        rotate_right(max(1, int((SPEED + pid_out) / 5)))
                        prev_left = prev_right = 0
                    elif Bias > -45:
                        move_right(max(1, int(SPEED / 15)))
                    else:
                        move_forward(SPEED)
                    time.sleep(0.001)

                else:
                    move_forward(SPEED)

            left_sum  = np.sum(histogram[:20])
            right_sum = np.sum(histogram[300:])
            if left_sum < right_sum:
                prev_left  += 1
            elif right_sum < left_sum:
                prev_right += 1


def main():
    key_thread  = threading.Thread(target=watch_for_q, daemon=True)
    follow_thread = threading.Thread(target=line_follow, daemon=True)

    key_thread.start()
    follow_thread.start()

    terminal_print("=== Camera Line Following ===")
    terminal_print("Press Q or Ctrl+C to stop.")

    try:
        while follow_thread.is_alive() and not quit_flag.is_set():
            time.sleep(0.1)
    except KeyboardInterrupt:
        terminal_print("\r\nCtrl+C pressed — stopping...")
        quit_flag.set()

    finally:
        quit_flag.set()
        follow_thread.join(timeout=2)
        stop_robot()
        bot.Ctrl_Ulatist_Switch(0)
        bot.Ctrl_BEEP_Switch(0)
        cap.release()
        del bot
        terminal_print("Stopped. Bot reset. Done.")


if __name__ == "__main__":
    main()