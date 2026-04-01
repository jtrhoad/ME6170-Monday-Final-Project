#!/usr/bin/env python3
"""
color_detection_test.py
=======================
Standalone color detection test using your desktop webcam.
No robot or hardware required.

WHAT IT DOES:
  - Opens your webcam
  - Detects black and white objects using HSV color masking
  - Draws bounding boxes and labels on detected regions
  - Displays live FPS

CONTROLS:
  Q  ->  quit
"""

import cv2
import time
import numpy as np

# --- Camera Settings ---
CAMERA_INDEX = 0      # 0 = default webcam. Try 1 or 2 if wrong camera opens.
FRAME_WIDTH  = 640
FRAME_HEIGHT = 480
TARGET_FPS   = 30

# --- Black HSV Range ---
# Black = any hue, any saturation, very low brightness (Value < 50)
# We ignore hue/saturation entirely -- black is just the absence of light.
BLACK_LOWER = np.array([0,   0,   0])
BLACK_UPPER = np.array([180, 255, 50])

# --- White HSV Range ---
# White = any hue, low saturation, high brightness
# Saturation < 80 catches off-whites and slightly tinted surfaces.
# Value > 170 catches whites in slightly dimmer lighting.
# If you get too many false positives (light greys, pastels), lower
# saturation back toward 50 or raise value back toward 200.
WHITE_LOWER = np.array([0,   0,   170])
WHITE_UPPER = np.array([180, 80,  255])

# --- Red HSV Range ---
# Red wraps around the hue boundary in OpenCV (0-180 scale).
# It appears at BOTH ends: 0-10 (lower red) and 170-180 (upper red).
# We define two ranges and combine them with a bitwise OR in detect_color.
RED_LOWER_1 = np.array([0,   80,  40])
RED_UPPER_1 = np.array([10,  255, 255])
RED_LOWER_2 = np.array([170, 80,  40])
RED_UPPER_2 = np.array([180, 255, 255])

# --- Orange HSV Range ---
# Orange sits cleanly between red and yellow (hue 10-20). No wrapping needed.
ORANGE_LOWER = np.array([10,  80,  80])
ORANGE_UPPER = np.array([20,  255, 255])

# --- Yellow HSV Range ---
# Yellow hue sits around 20-35 in OpenCV HSV.
YELLOW_LOWER = np.array([20,  80,  80])
YELLOW_UPPER = np.array([35,  255, 255])

# --- Green HSV Range ---
# Green hue sits around 40-80. Wide range to catch lime through forest green.
GREEN_LOWER = np.array([40,  80,  40])
GREEN_UPPER = np.array([80,  255, 255])

# --- Purple HSV Range ---
# Purple/violet hue sits around 130-160.
PURPLE_LOWER = np.array([130, 50,  40])
PURPLE_UPPER = np.array([160, 255, 255])

# --- Blue HSV Range ---
# Blue hue sits around 100-130 in OpenCV's HSV scale (which runs 0-180, not 0-360).
# Saturation > 80 ensures we only catch vivid blues, not washed-out grey-blues.
# Value > 40 excludes near-black dark blues.
BLUE_LOWER = np.array([100, 80,  40])
BLUE_UPPER = np.array([130, 255, 255])

# Minimum contour area in pixels^2 -- filters out noise and small shadows
MIN_CONTOUR_AREA = 800

# Max objects to track per color -- focuses detection on the largest (closest) object.
# Increase to 2-3 to track a couple foreground objects. Set to None for unlimited.
MAX_TARGETS = 1


def detect_color(frame, lower, upper, label, box_color,
                 lower2=None, upper2=None, max_targets=1):
    """
    Detect objects of a given HSV color range in a frame.

    Args:
        frame:       BGR image from cv2.VideoCapture
        lower:       np.array HSV lower bound
        upper:       np.array HSV upper bound
        label:       string shown on bounding box
        box_color:   BGR tuple for the bounding box color
        lower2:      optional second lower bound (for red hue wrapping)
        upper2:      optional second upper bound (for red hue wrapping)
        max_targets: how many objects to draw, largest first (default 1)
                     Set to None to draw all detected objects.

    WHY SORT BY AREA FOR FOREGROUND FOCUS?
        Closer objects appear larger in frame. Sorting contours by area
        descending and keeping only the top N naturally focuses on whatever
        is nearest to the camera -- the foreground object.

    WHY HSV INSTEAD OF BGR?
        BGR mixes color and brightness together, so "dark red" and "bright red"
        look completely different numerically. HSV separates Hue (color),
        Saturation (intensity), and Value (brightness), making it much easier
        to define a color range that holds up under different lighting.

    WHY TWO RANGES FOR RED?
        OpenCV's hue scale runs 0-180. Red sits at both ends (0-10 and 170-180)
        because hue is circular -- 180 wraps back to 0. A single inRange call
        can't straddle that boundary, so we create two masks and OR them together.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, lower, upper)

    # If a second range is provided, combine both masks
    if lower2 is not None and upper2 is not None:
        mask2 = cv2.inRange(hsv, lower2, upper2)
        mask = cv2.bitwise_or(mask, mask2)

    # Morphological cleanup:
    #   CLOSE - fills small holes inside detected blobs
    #   OPEN  - removes small noise specks outside blobs
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    # Filter noise, sort largest first, then limit to max_targets
    contours = [c for c in contours if cv2.contourArea(c) >= MIN_CONTOUR_AREA]
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    if max_targets is not None:
        contours = contours[:max_targets]

    count = 0
    for contour in contours:

        count += 1
        x, y, w, h = cv2.boundingRect(contour)
        cx, cy = x + w // 2, y + h // 2

        # Bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)

        # Label with filled background for readability
        font = cv2.FONT_HERSHEY_SIMPLEX
        (lw, lh), _ = cv2.getTextSize(label, font, 0.6, 2)
        cv2.rectangle(frame, (x, y - lh - 8), (x + lw + 4, y), box_color, -1)
        cv2.putText(frame, label, (x + 2, y - 4), font, 0.6, (255, 255, 255), 2)

        # Center dot
        cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)

    return frame, count


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          TARGET_FPS)

    if not cap.isOpened():
        print(f'ERROR: Could not open camera at index {CAMERA_INDEX}.')
        print('Try changing CAMERA_INDEX to 1 or 2 at the top of the script.')
        return

    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f'Camera opened: {FRAME_WIDTH}x{FRAME_HEIGHT} @ {actual_fps}fps')
    print('Press Q to quit.')

    # FPS tracking
    fps_counter = 0
    fps_display = 0
    fps_timer   = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print('WARNING: Failed to grab frame.')
            break

        # --- Run detection ---
        # Each color is one function call -- easy to add more later
        frame, red_count = detect_color(
            frame,
            RED_LOWER_1, RED_UPPER_1,
            label='Red',
            box_color=(0, 0, 200), max_targets=MAX_TARGETS,
            lower2=RED_LOWER_2, upper2=RED_UPPER_2
        )
        frame, orange_count = detect_color(
            frame,
            ORANGE_LOWER, ORANGE_UPPER,
            label='Orange',
            box_color=(0, 130, 255), max_targets=MAX_TARGETS
        )
        frame, black_count = detect_color(
            frame,
            BLACK_LOWER, BLACK_UPPER,
            label='Black',
            box_color=(50, 50, 50), max_targets=MAX_TARGETS
        )
        frame, white_count = detect_color(
            frame,
            WHITE_LOWER, WHITE_UPPER,
            label='White',
            box_color=(200, 200, 200), max_targets=MAX_TARGETS
        )
        frame, blue_count = detect_color(
            frame,
            BLUE_LOWER, BLUE_UPPER,
            label='Blue',
            box_color=(255, 100, 0), max_targets=MAX_TARGETS
        )
        frame, yellow_count = detect_color(
            frame,
            YELLOW_LOWER, YELLOW_UPPER,
            label='Yellow',
            box_color=(0, 215, 255), max_targets=MAX_TARGETS
        )
        frame, green_count = detect_color(
            frame,
            GREEN_LOWER, GREEN_UPPER,
            label='Green',
            box_color=(0, 200, 0), max_targets=MAX_TARGETS
        )
        frame, purple_count = detect_color(
            frame,
            PURPLE_LOWER, PURPLE_UPPER,
            label='Purple',
            box_color=(200, 0, 200), max_targets=MAX_TARGETS
        )

        # --- FPS counter ---
        fps_counter += 1
        if time.time() - fps_timer >= 1.0:
            fps_display = fps_counter
            fps_counter = 0
            fps_timer   = time.time()

        # --- HUD overlay ---
        cv2.putText(frame, f'FPS: {fps_display}',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f'Red objects:    {red_count}',
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 200), 2)
        cv2.putText(frame, f'Orange objects: {orange_count}',
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 130, 255), 2)
        cv2.putText(frame, f'Yellow objects: {yellow_count}',
                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 215, 255), 2)
        cv2.putText(frame, f'Green objects:  {green_count}',
                    (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
        cv2.putText(frame, f'Blue objects:   {blue_count}',
                    (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2)
        cv2.putText(frame, f'Purple objects: {purple_count}',
                    (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 0, 200), 2)
        cv2.putText(frame, f'White objects:  {white_count}',
                    (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        cv2.putText(frame, f'Black objects:  {black_count}',
                    (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 80), 2)
        cv2.putText(frame, 'Q: quit',
                    (10, FRAME_HEIGHT - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow('Color Detection Test', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print('Done.')


if __name__ == '__main__':
    main()