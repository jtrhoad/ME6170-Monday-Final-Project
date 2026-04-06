#!/usr/bin/env python3
"""
color_tracker.py
================
Combines multi-color detection with pan/tilt servo tracking.

MODES:
  Default (key 0) -- Detects all colors simultaneously. No servo movement.
  Single color     -- Tracks one color; servo centers on the largest (closest) object.

KEY BINDINGS:
  0  ->  All colors (default, no tracking)
  1  ->  Track Red
  2  ->  Track Orange
  3  ->  Track Yellow
  4  ->  Track Green
  5  ->  Track Blue
  6  ->  Track Purple
  7  ->  Track White
  8  ->  Track Black
  Q  ->  Quit

TRACKING DESIGN:
  - Proportional control: correction = gain * pixel_error
  - Max step per frame caps how fast the servo moves (smoothness)
  - Deadband: errors smaller than DEADBAND_PX are ignored (stops jitter)
  - Hard bounds: servo angles are clamped before every command (safety)
"""

import cv2
import time
import numpy as np
from Raspbot_Lib import Raspbot

# ---------------------------------------------------------------------------
# Servo Constants
# ---------------------------------------------------------------------------
SERVO_PAN   = 1
SERVO_TILT  = 2
PAN_CENTER  = 72
TILT_CENTER = 25
PAN_MIN,  PAN_MAX  =   5, 175   # Hard limits -- robot will not exceed these
TILT_MIN, TILT_MAX =   5,  95

# ---------------------------------------------------------------------------
# Tracking Tuning
# ---------------------------------------------------------------------------
# How many degrees to move per pixel of error.
# Lower = smoother but slower to acquire. Higher = faster but more jitter.
PAN_GAIN  = 0.04   # degrees per pixel (horizontal)
TILT_GAIN = 0.04   # degrees per pixel (vertical)

# Maximum servo movement per frame -- prevents lurching on large sudden errors.
MAX_STEP_DEG = 3.0

# If the object center is within this many pixels of frame center, don't move.
# This is the deadband -- eliminates constant micro-corrections when centered.
DEADBAND_PX = 25

# ---------------------------------------------------------------------------
# Camera Constants
# ---------------------------------------------------------------------------
CAMERA_INDEX = 0
FRAME_WIDTH  = 640
FRAME_HEIGHT = 480
FRAME_CX     = FRAME_WIDTH  // 2   # 320 -- horizontal center
FRAME_CY     = FRAME_HEIGHT // 2   # 240 -- vertical center
TARGET_FPS   = 30

# ---------------------------------------------------------------------------
# Detection Constants
# ---------------------------------------------------------------------------
MIN_CONTOUR_AREA = 800   # px^2 -- filters camera noise and tiny shadows

# ---------------------------------------------------------------------------
# HSV Color Ranges
# ---------------------------------------------------------------------------
# OpenCV HSV: Hue 0-180, Saturation 0-255, Value 0-255
# (Note: standard HSV is 0-360 hue; OpenCV halves it to fit in 8 bits)

# Red wraps around the hue boundary (0 and 180 are the same color).
# We need TWO ranges and combine them with bitwise OR -- see detect_color().
RED_LOWER_1 = np.array([0,   80,  40]);  RED_UPPER_1 = np.array([10,  255, 255])
RED_LOWER_2 = np.array([170, 80,  40]);  RED_UPPER_2 = np.array([180, 255, 255])

ORANGE_LOWER = np.array([10,  80,  80]);  ORANGE_UPPER = np.array([20,  255, 255])
YELLOW_LOWER = np.array([20,  80,  80]);  YELLOW_UPPER = np.array([35,  255, 255])
GREEN_LOWER  = np.array([40,  80,  40]);  GREEN_UPPER  = np.array([80,  255, 255])
BLUE_LOWER   = np.array([100, 80,  40]);  BLUE_UPPER   = np.array([130, 255, 255])
PURPLE_LOWER = np.array([130, 50,  40]);  PURPLE_UPPER = np.array([160, 255, 255])

# White and Black are special -- color (hue) doesn't define them, brightness does.
WHITE_LOWER  = np.array([0,   0,   170]); WHITE_UPPER  = np.array([180, 80,  255])
BLACK_LOWER  = np.array([0,   0,   0]);   BLACK_UPPER  = np.array([180, 255, 50])

# ---------------------------------------------------------------------------
# Color Registry
# Keys 1-8 map to: (label, box_color_BGR, lower, upper, lower2, upper2)
# lower2/upper2 are only needed for red's hue-wrap; None otherwise.
# ---------------------------------------------------------------------------
COLORS = {
    1: ("Red",    (0,   0,   200), RED_LOWER_1,  RED_UPPER_1,  RED_LOWER_2,  RED_UPPER_2),
    2: ("Orange", (0,   130, 255), ORANGE_LOWER, ORANGE_UPPER, None,         None),
    3: ("Yellow", (0,   215, 255), YELLOW_LOWER, YELLOW_UPPER, None,         None),
    4: ("Green",  (0,   200, 0),   GREEN_LOWER,  GREEN_UPPER,  None,         None),
    5: ("Blue",   (255, 100, 0),   BLUE_LOWER,   BLUE_UPPER,   None,         None),
    6: ("Purple", (200, 0,   200), PURPLE_LOWER, PURPLE_UPPER, None,         None),
    7: ("White",  (200, 200, 200), WHITE_LOWER,  WHITE_UPPER,  None,         None),
    8: ("Black",  (50,  50,  50),  BLACK_LOWER,  BLACK_UPPER,  None,         None),
}

# HUD y-positions for each color's object count (default mode display)
HUD_Y_POSITIONS = {1: 60, 2: 90, 3: 120, 4: 150, 5: 180, 6: 210, 7: 240, 8: 270}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def center_camera(bot):
    """
    Approach center from consistent directions to eliminate mechanical hysteresis.
    Hysteresis in servo gears means the final position depends on which direction
    you arrived from. By always approaching from the same direction, we get a
    repeatable resting position.
    """
    bot.Ctrl_Servo(SERVO_PAN,  180);       time.sleep(0.5)
    bot.Ctrl_Servo(SERVO_PAN,  PAN_CENTER); time.sleep(0.5)
    bot.Ctrl_Servo(SERVO_TILT, 0);         time.sleep(0.5)
    bot.Ctrl_Servo(SERVO_TILT, TILT_CENTER); time.sleep(0.5)
    print(f"Camera centered: pan={PAN_CENTER}, tilt={TILT_CENTER}")


def clamp(value, lo, hi):
    """Clamp value between lo and hi. Used on every servo command."""
    return max(lo, min(hi, value))


def compute_servo_step(error_px, gain, max_step):
    """
    Convert a pixel error into a servo degree step.

    1. Scale by gain (error -> degrees).
    2. Cap at max_step so large errors don't cause sudden lurches.
    3. Preserve sign so direction is correct.

    Example: error=80px, gain=0.04, max_step=3.0
      raw = 80 * 0.04 = 3.2 -> clamped to 3.0 degrees
    """
    raw = error_px * gain
    return float(np.clip(raw, -max_step, max_step))


def detect_color(frame, lower, upper, label, box_color,
                 lower2=None, upper2=None, max_targets=None):
    """
    Detect colored objects in frame. Returns annotated frame and a list of
    (cx, cy, area) tuples for each detected object, sorted largest first.

    Args:
        frame      -- BGR image
        lower/upper -- HSV bounds for the primary range
        label      -- text shown on bounding box
        box_color  -- BGR color for the box
        lower2/upper2 -- optional second HSV range (red hue-wrap)
        max_targets -- max objects to annotate (None = all)

    Returns:
        frame  -- annotated image
        targets -- list of (center_x, center_y, area) sorted by area desc
    """
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)

    if lower2 is not None and upper2 is not None:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower2, upper2))

    # Morphological cleanup:
    #   CLOSE -- fills small holes inside blobs (e.g., reflections on a surface)
    #   OPEN  -- removes small speckle noise outside blobs
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= MIN_CONTOUR_AREA]
    contours = sorted(contours, key=cv2.contourArea, reverse=True)  # largest first

    if max_targets is not None:
        contours = contours[:max_targets]

    targets = []
    font = cv2.FONT_HERSHEY_SIMPLEX

    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        cx, cy = x + w // 2, y + h // 2
        targets.append((cx, cy, area))

        # Bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)

        # Label with filled background
        (lw, lh), _ = cv2.getTextSize(label, font, 0.6, 2)
        cv2.rectangle(frame, (x, y - lh - 8), (x + lw + 4, y), box_color, -1)
        cv2.putText(frame, label, (x + 2, y - 4), font, 0.6, (255, 255, 255), 2)

        # Center dot -- yellow for visibility
        cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)

    return frame, targets


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    robot = Raspbot()
    center_camera(robot)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          TARGET_FPS)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))

    if not cap.isOpened():
        print("ERROR: Could not open camera.")
        del robot
        return

    print(f"Camera opened: {FRAME_WIDTH}x{FRAME_HEIGHT} @ {cap.get(cv2.CAP_PROP_FPS)}fps")
    print("Keys: 0=all colors  1=Red  2=Orange  3=Yellow  4=Green")
    print("      5=Blue  6=Purple  7=White  8=Black  Q=quit")

    # --- State ---
    tracking_mode = 0       # 0 = show all; 1-8 = track that color
    pan           = PAN_CENTER
    tilt          = TILT_CENTER
    last_pan      = PAN_CENTER
    last_tilt     = TILT_CENTER

    # FPS tracking
    fps_counter = 0
    fps_display = 0
    fps_timer   = time.time()

    font = cv2.FONT_HERSHEY_SIMPLEX

    while True:
        ret, frame = cap.read()
        if not ret:
            print("WARNING: Failed to grab frame.")
            break

        # ------------------------------------------------------------------
        # Detection
        # ------------------------------------------------------------------
        if tracking_mode == 0:
            # --- Default mode: annotate all colors, no servo movement ---
            counts = {}
            for key, (label, color, lo, hi, lo2, hi2) in COLORS.items():
                frame, targets = detect_color(frame, lo, hi, label, color,
                                              lo2, hi2, max_targets=None)
                counts[key] = len(targets)

        else:
            # --- Single-color tracking mode ---
            label, box_color, lo, hi, lo2, hi2 = COLORS[tracking_mode]
            frame, targets = detect_color(frame, lo, hi, label, box_color,
                                          lo2, hi2, max_targets=1)

            if targets:
                # targets[0] is the largest (closest) object
                obj_cx, obj_cy, _ = targets[0]

                # Pixel error = how far the object is from frame center
                # Positive x_err -> object is RIGHT of center -> decrease pan to follow
                # Positive y_err -> object is BELOW center  -> decrease tilt to follow
                x_err = obj_cx - FRAME_CX
                y_err = obj_cy - FRAME_CY

                # Draw crosshair at frame center for reference
                cv2.line(frame, (FRAME_CX - 15, FRAME_CY),
                                (FRAME_CX + 15, FRAME_CY), (0, 255, 0), 1)
                cv2.line(frame, (FRAME_CX, FRAME_CY - 15),
                                (FRAME_CX, FRAME_CY + 15), (0, 255, 0), 1)

                # Draw error line from frame center to object center
                cv2.line(frame, (FRAME_CX, FRAME_CY),
                                (obj_cx, obj_cy), (0, 200, 255), 1)

                # --- Proportional servo control with deadband ---
                #
                # Deadband: if the error is small, don't move.
                # This prevents the servo from hunting back and forth when
                # the object is already nearly centered.
                #
                # Outside the deadband, scale the error into a degree correction,
                # then clamp it to MAX_STEP_DEG so we never lurch.

                if abs(x_err) > DEADBAND_PX:
                    pan = clamp(pan - compute_servo_step(x_err, PAN_GAIN, MAX_STEP_DEG),
                                PAN_MIN, PAN_MAX)

                if abs(y_err) > DEADBAND_PX:
                    tilt = clamp(tilt - compute_servo_step(y_err, TILT_GAIN, MAX_STEP_DEG),
                                 TILT_MIN, TILT_MAX)

        # ------------------------------------------------------------------
        # Servo Commands (only send if angle actually changed)
        # ------------------------------------------------------------------
        pan_int  = int(round(pan))
        tilt_int = int(round(tilt))

        if pan_int != last_pan:
            robot.Ctrl_Servo(SERVO_PAN, pan_int)
            last_pan = pan_int

        if tilt_int != last_tilt:
            robot.Ctrl_Servo(SERVO_TILT, tilt_int)
            last_tilt = tilt_int

        # ------------------------------------------------------------------
        # FPS
        # ------------------------------------------------------------------
        fps_counter += 1
        if time.time() - fps_timer >= 1.0:
            fps_display = fps_counter
            fps_counter = 0
            fps_timer   = time.time()

        # ------------------------------------------------------------------
        # HUD Overlay
        # ------------------------------------------------------------------
        mode_label = "ALL COLORS" if tracking_mode == 0 else f"TRACKING: {COLORS[tracking_mode][0].upper()}"
        cv2.putText(frame, f"FPS: {fps_display}  |  Mode: {mode_label}",
                    (10, 30), font, 0.65, (0, 255, 0), 2)
        cv2.putText(frame, f"Pan: {pan_int}  Tilt: {tilt_int}",
                    (10, FRAME_HEIGHT - 35), font, 0.55, (200, 200, 200), 1)
        cv2.putText(frame, "0:all  1:Red 2:Org 3:Yel 4:Grn 5:Blu 6:Pur 7:Wht 8:Blk  Q:quit",
                    (10, FRAME_HEIGHT - 15), font, 0.45, (180, 180, 180), 1)

        if tracking_mode == 0:
            # Show per-color counts on the left side
            for key, count in counts.items():
                label = COLORS[key][0]
                color = COLORS[key][1]
                y     = HUD_Y_POSITIONS[key]
                cv2.putText(frame, f"{label}: {count}",
                            (10, y), font, 0.6, color, 2)

        # ------------------------------------------------------------------
        # Key Handling
        # ------------------------------------------------------------------
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('0'):
            tracking_mode = 0
            # Return camera to center when exiting tracking mode
            center_camera(robot)
            pan  = PAN_CENTER
            tilt = TILT_CENTER
        elif key in [ord(str(k)) for k in range(1, 9)]:
            tracking_mode = int(chr(key))
            print(f"Tracking: {COLORS[tracking_mode][0]}")

        cv2.imshow("Raspbot Color Tracker", frame)

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    del robot
    print("Stream closed.")


if __name__ == "__main__":
    main()
