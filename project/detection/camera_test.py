#!/usr/bin/env python3
"""
camera_test.py
Mouse-controlled PTZ camera stream.
- Click and drag to pan/tilt
- Scroll wheel to digital zoom
- Press q to quit
"""

import cv2
import time
import numpy as np
from Raspbot_Lib import Raspbot

# --- Constants ---
SERVO_PAN   = 1
SERVO_TILT  = 2
PAN_CENTER  = 72
TILT_CENTER = 25
PAN_MIN,  PAN_MAX  = 0,   180
TILT_MIN, TILT_MAX = 0,   100

CAMERA_INDEX = 0
FRAME_WIDTH  = 640
FRAME_HEIGHT = 480
TARGET_FPS   = 30

# --- State ---
pan        = PAN_CENTER
tilt       = TILT_CENTER
last_pan   = PAN_CENTER
last_tilt  = TILT_CENTER
dragging   = False
drag_start = (0, 0)
zoom_level = 1.0   # 1.0 = no zoom, 2.0 = 2x digital zoom
robot      = None


def center_camera(bot):
    """Approach center from consistent directions to eliminate hysteresis."""
    bot.Ctrl_Servo(SERVO_PAN, 180)
    time.sleep(0.5)
    bot.Ctrl_Servo(SERVO_PAN, PAN_CENTER)
    time.sleep(0.5)
    bot.Ctrl_Servo(SERVO_TILT, 0)
    time.sleep(0.5)
    bot.Ctrl_Servo(SERVO_TILT, TILT_CENTER)
    time.sleep(0.5)
    print(f'Camera centered: pan={PAN_CENTER}, tilt={TILT_CENTER}')


def mouse_callback(event, x, y, flags, param):
    """
    Left click + drag  → pan and tilt the camera
    Scroll wheel up    → zoom in
    Scroll wheel down  → zoom out
    """
    global pan, tilt, dragging, drag_start, zoom_level

    # --- Left button pressed: start drag ---
    if event == cv2.EVENT_LBUTTONDOWN:
        dragging = True
        drag_start = (x, y)

    # --- Mouse moving while held: calculate pan/tilt delta ---
    elif event == cv2.EVENT_MOUSEMOVE and dragging:
        dx = x - drag_start[0]   # horizontal movement → pan
        dy = y - drag_start[1]   # vertical movement   → tilt

        # Scale drag pixels to servo degrees
        # Dividing by 8 gives a comfortable sensitivity
        pan  = int(np.clip(pan  + dx / 8, PAN_MIN,  PAN_MAX))
        tilt = int(np.clip(tilt + dy / 8, TILT_MIN, TILT_MAX))

        # Reset drag start so movement is incremental not absolute
        drag_start = (x, y)

    # --- Left button released: stop drag ---
    elif event == cv2.EVENT_LBUTTONUP:
        dragging = False

    # --- Scroll wheel: digital zoom ---
    elif event == cv2.EVENT_MOUSEWHEEL:
        if flags > 0:   # scroll up = zoom in
            zoom_level = min(3.0, zoom_level + 0.1)
        else:           # scroll down = zoom out
            zoom_level = max(1.0, zoom_level - 0.1)


def apply_zoom(frame, zoom):
    """Crop and resize frame to simulate digital zoom."""
    if zoom == 1.0:
        return frame
    h, w = frame.shape[:2]
    # Calculate crop box centered on frame
    new_h = int(h / zoom)
    new_w = int(w / zoom)
    y1 = (h - new_h) // 2
    x1 = (w - new_w) // 2
    cropped = frame[y1:y1+new_h, x1:x1+new_w]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)


def main():
    global pan, tilt, last_pan, last_tilt, robot

    robot = Raspbot()
    center_camera(robot)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          TARGET_FPS)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M','J','P','G'))

    if not cap.isOpened():
        print('ERROR: Could not open camera.')
        del robot
        return

    cv2.namedWindow('Raspbot Camera Stream')
    cv2.setMouseCallback('Raspbot Camera Stream', mouse_callback)

    print('Controls:')
    print('  Click + drag  → pan/tilt camera')
    print('  Scroll wheel  → zoom in/out')
    print('  Q             → quit')

    while True:
        ret, frame = cap.read()
        if not ret:
            print('WARNING: Failed to grab frame.')
            break

        # --- Send servo commands only when values change ---
        if pan != last_pan:
            robot.Ctrl_Servo(SERVO_PAN, pan)
            last_pan = pan
        if tilt != last_tilt:
            robot.Ctrl_Servo(SERVO_TILT, tilt)
            last_tilt = tilt

        # --- Apply digital zoom ---
        frame = apply_zoom(frame, zoom_level)

        # --- HUD overlay ---
        cv2.putText(frame, f'Pan: {pan}  Tilt: {tilt}  Zoom: {zoom_level:.1f}x',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)
        cv2.putText(frame, 'Drag to pan/tilt | Scroll to zoom | Q to quit',
                    (10, FRAME_HEIGHT - 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (200, 200, 200), 1)

        cv2.imshow('Raspbot Camera Stream', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    del robot
    print('Stream closed.')


if __name__ == '__main__':
    main()