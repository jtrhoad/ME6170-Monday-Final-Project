#!/usr/bin/env python3
"""
camera_test.py
Streams live video with an OpenCV control panel for real-time
pan/tilt adjustment. Press 'q' to quit.
"""

import cv2
import time
from Raspbot_Lib import Raspbot

# --- Constants ---
SERVO_PAN   = 1
SERVO_TILT  = 2
PAN_CENTER  = 72    # calibrated approaching from 180
TILT_CENTER = 25    # calibrated approaching from 0
PAN_MIN, PAN_MAX   = 0, 180
TILT_MIN, TILT_MAX = 0, 100

CAMERA_INDEX = 0
FRAME_WIDTH  = 640
FRAME_HEIGHT = 480
TARGET_FPS   = 30

# Track last sent servo values to avoid spamming I2C
last_pan  = PAN_CENTER
last_tilt = TILT_CENTER


def center_camera(robot):
    """Approach center from consistent directions to eliminate hysteresis."""
    robot.Ctrl_Servo(SERVO_PAN, 180)
    time.sleep(0.5)
    robot.Ctrl_Servo(SERVO_PAN, PAN_CENTER)
    time.sleep(0.5)
    robot.Ctrl_Servo(SERVO_TILT, 0)
    time.sleep(0.5)
    robot.Ctrl_Servo(SERVO_TILT, TILT_CENTER)
    time.sleep(0.5)
    print(f'Camera centered: pan={PAN_CENTER}, tilt={TILT_CENTER}')


def nothing(x):
    """Required placeholder callback for cv2.createTrackbar."""
    pass


def main():
    global last_pan, last_tilt

    robot = Raspbot()
    center_camera(robot)

    # --- Camera setup ---
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          TARGET_FPS)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M','J','P','G'))

    if not cap.isOpened():
        print('ERROR: Could not open camera.')
        del robot
        return

    # --- Control panel window with trackbars ---
    cv2.namedWindow('Camera Controls')
    cv2.createTrackbar('Pan  (L <-> R)', 'Camera Controls',
                       PAN_CENTER,  PAN_MAX,  nothing)
    cv2.createTrackbar('Tilt (D <-> U)', 'Camera Controls',
                       TILT_CENTER, TILT_MAX, nothing)

    print('Controls open. Use sliders to pan/tilt.')
    print('Keyboard: A/D = pan left/right | W/S = tilt up/down | Q = quit')

    while True:
        ret, frame = cap.read()
        if not ret:
            print('WARNING: Failed to grab frame.')
            break

        # --- Read trackbar values ---
        pan  = cv2.getTrackbarPos('Pan  (L <-> R)', 'Camera Controls')
        tilt = cv2.getTrackbarPos('Tilt (D <-> U)', 'Camera Controls')

        # --- Only send I2C command if value changed (avoid bus spam) ---
        if pan != last_pan:
            robot.Ctrl_Servo(SERVO_PAN, pan)
            last_pan = pan

        if tilt != last_tilt:
            robot.Ctrl_Servo(SERVO_TILT, tilt)
            last_tilt = tilt

        # --- Overlay current position on frame ---
        cv2.putText(frame, f'Pan: {pan}  Tilt: {tilt}',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)

        cv2.imshow('Raspbot Camera Stream', frame)

        # --- Keyboard controls (5 degrees per keypress) ---
        key = cv2.waitKey(1)   # No & 0xFF mask — arrow keys need full value
        if key == ord('q'):
            break
        elif key == 65361:   # ← Left arrow  = pan left
            new_pan = max(PAN_MIN, pan - 5)
            cv2.setTrackbarPos('Pan  (L <-> R)', 'Camera Controls', new_pan)
        elif key == 65363:   # → Right arrow = pan right
            new_pan = min(PAN_MAX, pan + 5)
            cv2.setTrackbarPos('Pan  (L <-> R)', 'Camera Controls', new_pan)
        elif key == 65362:   # ↑ Up arrow    = tilt up
            new_tilt = min(TILT_MAX, tilt + 5)
            cv2.setTrackbarPos('Tilt (D <-> U)', 'Camera Controls', new_tilt)
        elif key == 65364:   # ↓ Down arrow  = tilt down
            new_tilt = max(TILT_MIN, tilt - 5)
            cv2.setTrackbarPos('Tilt (D <-> U)', 'Camera Controls', new_tilt)

    # --- Cleanup ---
    cap.release()
    cv2.destroyAllWindows()
    del robot
    print('Stream closed.')


if __name__ == '__main__':
    main()