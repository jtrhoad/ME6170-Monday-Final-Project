#!/usr/bin/env python3
"""
camera_test.py
Mouse-controlled PTZ camera stream with red object detection.
- Click and drag to pan/tilt
- Scroll wheel to digital zoom
- Press q to quit
"""

import cv2
import time
import numpy as np
from Raspbot_Lib import Raspbot

# --- Servo Constants ---
SERVO_PAN   = 1
SERVO_TILT  = 2
PAN_CENTER  = 72
TILT_CENTER = 25
PAN_MIN,  PAN_MAX  =   5, 175
TILT_MIN, TILT_MAX =   5,  95

# --- Camera Constants ---
CAMERA_INDEX = 0
FRAME_WIDTH  = 640
FRAME_HEIGHT = 480
TARGET_FPS   = 60    # Request 60fps — actual rate depends on USB bandwidth

# --- Black HSV Range ---
# Black = any hue, any saturation, very low brightness
# Value threshold is the key — below ~50 is reliably black
BLACK_LOWER = np.array([0,   0,   0])
BLACK_UPPER = np.array([180, 255, 50])

MIN_CONTOUR_AREA = 800

# --- State ---
pan        = PAN_CENTER
tilt       = TILT_CENTER
last_pan   = PAN_CENTER
last_tilt  = TILT_CENTER
dragging   = False
drag_start = (0, 0)
zoom_level = 1.0
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


def detect_black(frame):
    """
    Detect black objects using HSV color masking.

    Why Value (brightness) only?
    Black has no meaningful hue or saturation — it's simply
    the absence of light. So we ignore hue entirely and only
    threshold on the V channel being very low (< 50).
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, BLACK_LOWER, BLACK_UPPER)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) < MIN_CONTOUR_AREA:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        # Bounding box — dark gray color
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 50), 2)

        # Label background
        label = 'Black Object'
        font  = cv2.FONT_HERSHEY_SIMPLEX
        (lw, lh), _ = cv2.getTextSize(label, font, 0.6, 2)
        cv2.rectangle(frame, (x, y - lh - 8), (x + lw + 4, y), (50, 50, 50), -1)

        # Label text
        cv2.putText(frame, label, (x + 2, y - 4),
                    font, 0.6, (255, 255, 255), 2)

        # Center dot — yellow for visibility against black
        cv2.circle(frame, (x + w // 2, y + h // 2), 4, (0, 255, 255), -1)

    return frame


def mouse_callback(event, x, y, flags, param):
    global pan, tilt, dragging, drag_start, zoom_level

    if event == cv2.EVENT_LBUTTONDOWN:
        dragging = True
        drag_start = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE and dragging:
        dx = x - drag_start[0]
        dy = y - drag_start[1]
        pan  = int(np.clip(pan  + dx / 8, PAN_MIN,  PAN_MAX))
        tilt = int(np.clip(tilt + dy / 8, TILT_MIN, TILT_MAX))
        drag_start = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        dragging = False

    elif event == cv2.EVENT_MOUSEWHEEL:
        if flags > 0:
            zoom_level = min(3.0, zoom_level + 0.1)
        else:
            zoom_level = max(1.0, zoom_level - 0.1)


def apply_zoom(frame, zoom):
    """Crop and resize frame to simulate digital zoom."""
    if zoom == 1.0:
        return frame
    h, w = frame.shape[:2]
    new_h = int(h / zoom)
    new_w = int(w / zoom)
    y1 = (h - new_h) // 2
    x1 = (w - new_w) // 2
    cropped = frame[y1:y1 + new_h, x1:x1 + new_w]
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

    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f'Camera opened at {FRAME_WIDTH}x{FRAME_HEIGHT} @ {actual_fps}fps')

    cv2.namedWindow('Raspbot Camera Stream')
    cv2.setMouseCallback('Raspbot Camera Stream', mouse_callback)

    # FPS counter variables
    fps_counter = 0
    fps_display = 0
    fps_timer   = time.time()

    print('Controls:')
    print('  Click + drag  → pan/tilt camera')
    print('  Scroll wheel  → zoom in/out')
    print('  Q             → quit')

    while True:
        ret, frame = cap.read()
        if not ret:
            print('WARNING: Failed to grab frame.')
            break

        # --- Servo updates ---
        if pan != last_pan:
            robot.Ctrl_Servo(SERVO_PAN, pan)
            last_pan = pan
        if tilt != last_tilt:
            robot.Ctrl_Servo(SERVO_TILT, tilt)
            last_tilt = tilt

        # --- Apply zoom ---
        frame = apply_zoom(frame, zoom_level)

        # --- Run color detection ---
        frame = detect_red(frame)

        # --- Calculate actual FPS ---
        fps_counter += 1
        elapsed = time.time() - fps_timer
        if elapsed >= 1.0:
            fps_display = fps_counter
            fps_counter = 0
            fps_timer   = time.time()

        # --- HUD ---
        cv2.putText(frame,
                    f'Pan:{pan} Tilt:{tilt} Zoom:{zoom_level:.1f}x FPS:{fps_display}',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 255, 0), 2)
        cv2.putText(frame, 'Drag:pan/tilt  Scroll:zoom  Q:quit',
                    (10, FRAME_HEIGHT - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow('Raspbot Camera Stream', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    del robot
    print('Stream closed.')


if __name__ == '__main__':
    main()
```

---

## What Was Added

**Red detection pipeline** — each frame goes through this process:
```
BGR frame → convert to HSV
→ mask1 (hue 0-10)  + mask2 (hue 170-180)
→ combine masks
→ clean up noise (morphology)
→ find contours
→ draw box + label on any contour > 800px²