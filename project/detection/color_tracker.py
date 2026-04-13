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
# Tracking Tuning -- PID controllers for pan and tilt
# ---------------------------------------------------------------------------
# Servo control runs as PID on the pixel error between the object and the
# frame center. The output is a per-frame degree step that gets added to the
# current servo angle.
#
#   Kp -- proportional: drives the servo toward the target. Bigger = faster
#         acquisition but more overshoot/oscillation.
#   Ki -- integral: eliminates steady-state offset. KEEP THIS SMALL on a
#         servo system because the servo itself acts as an integrator
#         (each step accumulates into position). Too much Ki -> windup
#         and slow oscillation.
#   Kd -- derivative: damps overshoot by reacting to how fast the error
#         is changing. Helps a lot for smooth lock, but amplifies noise
#         from the camera. We tame that noise with an EMA filter on the
#         error signal (see ERROR_EMA_ALPHA below).
#
# Tuning recipe if you want to adjust:
#   1. Set Ki=0, Kd=0. Raise Kp until the camera tracks but oscillates a bit.
#   2. Add Kd until the oscillation damps out (start at Kp/3).
#   3. Add a tiny Ki only if you see a persistent off-center offset.
PAN_KP   = 0.040
PAN_KI   = 0.000
PAN_KD   = 0.012

TILT_KP  = 0.040
TILT_KI  = 0.000
TILT_KD  = 0.012

# Per-frame max servo step (degrees). Caps lurching on big sudden errors.
MAX_STEP_DEG = 2.5

# Anti-windup clamp on the integral term. Has no effect when Ki=0.
PID_INTEGRAL_LIMIT = 500.0

# Deadband: errors smaller than this many pixels are treated as zero.
# Wide enough to absorb residual overshoot so the servo can settle
# instead of ping-ponging back and forth.
DEADBAND_PX = 12

# Error smoothing -- exponential moving average on the pixel error.
# Camera detection coordinates jitter by a few pixels frame-to-frame even
# when nothing is moving. Feeding raw noisy error into a derivative term
# produces violent twitches. The EMA low-pass-filters the error signal
# before it reaches the PID:
#   alpha = 1.0 -> no smoothing (raw error)
#   alpha = 0.3 -> blends 30% new + 70% old, noticeable smoothing
#   alpha = 0.1 -> heavy smoothing, slower response
ERROR_EMA_ALPHA = 0.35

# ---------------------------------------------------------------------------
# Camera Constants
# ---------------------------------------------------------------------------
CAMERA_INDEX = 0
FRAME_WIDTH  = 640
FRAME_HEIGHT = 480
FRAME_CX     = FRAME_WIDTH  // 2   # 320 -- horizontal center
FRAME_CY     = FRAME_HEIGHT // 2   # 240 -- vertical center
FRAME_AREA   = FRAME_WIDTH * FRAME_HEIGHT
TARGET_FPS   = 60

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


# ---------------------------------------------------------------------------
# PID Controller
# ---------------------------------------------------------------------------

class PIDController:
    """
    Discrete-time PID with anti-windup, output clamping, and a deadband.

    Used here to convert pixel error -> servo degree step.
        update(error_px) -> step_deg

    KEY DESIGN CHOICES:
      - Time-aware: dt is measured between calls, so the controller behaves
        consistently even if the loop frame rate varies.
      - Anti-windup: the integral term is clamped, so a long burst of large
        error doesn't accumulate authority that overshoots later.
      - Deadband applied to error before all terms: tiny errors become 0,
        which kills jitter when the object is already near center.
      - Output clamp: caps the per-frame step so the servo never lurches.
    """

    def __init__(self, kp, ki, kd,
                 output_limit=None, integral_limit=None, deadband=0.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limit   = output_limit
        self.integral_limit = integral_limit
        self.deadband       = deadband
        self.reset()

    def reset(self):
        self._integral  = 0.0
        self._last_err  = 0.0
        self._last_time = None

    def update(self, error):
        now = time.time()
        dt  = 0.0 if self._last_time is None else (now - self._last_time)
        self._last_time = now

        # Deadband -- treat tiny errors as exactly zero
        if abs(error) < self.deadband:
            error = 0.0

        # Integral with anti-windup
        if dt > 0:
            self._integral += error * dt
            if self.integral_limit is not None:
                self._integral = float(np.clip(self._integral,
                                               -self.integral_limit,
                                                self.integral_limit))

        # Derivative
        derivative = 0.0
        if dt > 0:
            derivative = (error - self._last_err) / dt
        self._last_err = error

        output = self.kp * error + self.ki * self._integral + self.kd * derivative

        if self.output_limit is not None:
            output = float(np.clip(output, -self.output_limit, self.output_limit))
        return output


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
        targets -- list of (center_x, center_y, area, area_ratio) sorted by
                   area desc. area_ratio is area / FRAME_AREA -- the fraction
                   of the frame the object occupies, useful as a proxy for
                   distance (bigger = closer).
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
        area_ratio = area / FRAME_AREA
        x, y, w, h = cv2.boundingRect(contour)
        cx, cy = x + w // 2, y + h // 2
        targets.append((cx, cy, area, area_ratio))

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
    pan           = float(PAN_CENTER)
    tilt          = float(TILT_CENTER)
    last_pan      = PAN_CENTER
    last_tilt     = TILT_CENTER

    # PID controllers -- one per axis. Reset whenever the mode changes
    # or the target is lost so stale integral/derivative state can't kick
    # the camera on reacquisition.
    pan_pid = PIDController(
        kp=PAN_KP, ki=PAN_KI, kd=PAN_KD,
        output_limit=MAX_STEP_DEG, integral_limit=PID_INTEGRAL_LIMIT,
        deadband=DEADBAND_PX,
    )
    tilt_pid = PIDController(
        kp=TILT_KP, ki=TILT_KI, kd=TILT_KD,
        output_limit=MAX_STEP_DEG, integral_limit=PID_INTEGRAL_LIMIT,
        deadband=DEADBAND_PX,
    )

    # EMA-filtered error -- low-pass on the noisy pixel error signal.
    # None means "not initialized yet"; first valid sample seeds it.
    x_err_filt = None
    y_err_filt = None

    # Most recent target info, for HUD display
    current_area_ratio = 0.0
    target_visible     = False

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
            target_visible     = False
            current_area_ratio = 0.0

        else:
            # --- Single-color tracking mode ---
            label, box_color, lo, hi, lo2, hi2 = COLORS[tracking_mode]
            frame, targets = detect_color(frame, lo, hi, label, box_color,
                                          lo2, hi2, max_targets=1)

            if targets:
                # targets[0] is the largest (closest) object
                obj_cx, obj_cy, _, area_ratio = targets[0]
                current_area_ratio = area_ratio
                target_visible     = True

                # Pixel error = how far the object is from frame center
                # Positive x_err -> object is RIGHT of center
                # Positive y_err -> object is BELOW center
                x_err_raw = obj_cx - FRAME_CX
                y_err_raw = obj_cy - FRAME_CY

                # --- EMA low-pass on the error signal ---
                # Camera detection coords jitter by a few pixels even on a
                # static object. Filtering before the PID kills the jitter
                # without slowing down real motion much.
                if x_err_filt is None:
                    x_err_filt = float(x_err_raw)
                    y_err_filt = float(y_err_raw)
                else:
                    a = ERROR_EMA_ALPHA
                    x_err_filt = a * x_err_raw + (1 - a) * x_err_filt
                    y_err_filt = a * y_err_raw + (1 - a) * y_err_filt

                # Draw crosshair at frame center for reference
                cv2.line(frame, (FRAME_CX - 15, FRAME_CY),
                                (FRAME_CX + 15, FRAME_CY), (0, 255, 0), 1)
                cv2.line(frame, (FRAME_CX, FRAME_CY - 15),
                                (FRAME_CX, FRAME_CY + 15), (0, 255, 0), 1)

                # Draw error line from frame center to object center
                cv2.line(frame, (FRAME_CX, FRAME_CY),
                                (obj_cx, obj_cy), (0, 200, 255), 1)

                # --- PID servo control ---
                # PID outputs the per-frame degree step directly. The minus
                # sign maps "object is right -> servo pan angle decreases"
                # (or vice versa, depending on your servo wiring -- flip the
                # sign here if the camera moves the wrong way).
                pan_step  = pan_pid.update(x_err_filt)
                tilt_step = tilt_pid.update(y_err_filt)

                pan  = clamp(pan  - pan_step,  PAN_MIN,  PAN_MAX)
                tilt = clamp(tilt - tilt_step, TILT_MIN, TILT_MAX)

            else:
                # Target lost -- reset filters and PID state so we don't
                # snap when reacquired with stale derivative/integral.
                target_visible     = False
                current_area_ratio = 0.0
                x_err_filt         = None
                y_err_filt         = None
                pan_pid.reset()
                tilt_pid.reset()

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
        else:
            # Single-color tracking: show area ratio of the locked target
            track_color = COLORS[tracking_mode][1]
            if target_visible:
                area_text = f"Area: {current_area_ratio * 100:.2f}% of frame"
            else:
                area_text = "Area: -- (no target)"
            cv2.putText(frame, area_text,
                        (10, 60), font, 0.7, track_color, 2)

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
            pan  = float(PAN_CENTER)
            tilt = float(TILT_CENTER)
            pan_pid.reset()
            tilt_pid.reset()
            x_err_filt = None
            y_err_filt = None
        elif key in [ord(str(k)) for k in range(1, 9)]:
            tracking_mode = int(chr(key))
            pan_pid.reset()
            tilt_pid.reset()
            x_err_filt = None
            y_err_filt = None
            print(f"Tracking: {COLORS[tracking_mode][0]}")

        cv2.imshow("Raspbot Color Tracker", frame)

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    del robot
    print("Stream closed.")


if __name__ == "__main__":
    main()