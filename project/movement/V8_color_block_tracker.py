#!/usr/bin/env python3
"""
color_block_tracker.py
======================
Generalized color block tracker for Raspbot V2 with Mecanum wheels.

STARTUP:
  An on-screen color selection menu appears over the live camera feed.
  Press 1-8 to pick a color. The menu disappears and tracking begins.

STATE MACHINE:
  SEARCHING   -- Rotate bot to scan for target color blob.
  TRACKING    -- Blob found; rotate body + tilt camera to center it.
  CONFIRMING  -- Centered; ask user Y/N before approaching.
  APPROACHING -- Confirmed; drive forward (PID yaw) toward target.
  SCANNING_OBSTACLE -- Obstacle in path; rotate body L/R to find clearer side.
                 Re-scans prioritize the direction back toward the target.
  AVOIDING    -- Wall-following: drive forward, peek sideways to check if wall
                 has ended, loop until clear, strafe past the corner.
                 New obstacles ahead trigger a re-scan with target priority.
  ARRIVED     -- Within ARRIVAL_DISTANCE_CM of target; victory dance.

CONFIRMING / REJECTION:
  When the camera centers on a target, the bot stops and overlays a prompt.
  Y  ->  Confirm: advance to APPROACHING.
  N  ->  Reject: blacklist that screen region for BLACKLIST_DURATION seconds,
         then return to SEARCHING. The blacklist prevents immediate re-lock
         on the same object. Blacklisted zones are drawn on-screen as faint
         red circles with a countdown timer.

CONTROLS:
  1-8  ->  Select color (startup screen only)
  Y    ->  Confirm target  (CONFIRMING state only)
  N    ->  Reject target   (CONFIRMING state only)
  Q    ->  Quit

BEFORE RUNNING:
  Calibrate DEG_PER_SECOND_ROTATE and CM_PER_SECOND_FORWARD
  using dr_calibration.py on the actual surface.
"""

import cv2
import time
import math
import threading
import numpy as np
from Raspbot_Lib import Raspbot

# ===========================================================================
# DEBUG / LOGGING
# Setting DEBUG_PRINT = False silences all non-critical prints. Each print
# over an SSH or VNC terminal can take 1-3 ms and adds up over a frame
# loop -- removing them is one of the cheapest latency wins available.
# Critical messages (errors, state-change banners) still go through.
# ===========================================================================
DEBUG_PRINT = False

def dbg(msg):
    if DEBUG_PRINT:
        print(msg)


# ===========================================================================
# THREADED CAMERA READER
# Reads frames in a background thread and always holds the latest one.
# The main loop calls .read() to get whatever is currently in the slot --
# it never blocks waiting for a new frame.
#
# WHY THIS HELPS:
#   - cap.read() is a blocking call; if the camera is mid-exposure when you
#     call it, the main loop just sits there. With a threaded reader, the
#     main loop always returns instantly with whatever the latest frame is.
#   - More importantly, when a state runner calls time.sleep() (e.g., for a
#     rotation step), the camera buffer fills with stale frames. The threaded
#     reader continuously drains the buffer in the background, so when the
#     main loop resumes it gets a *current* frame instead of a 200ms-old one.
#
# COST: one extra thread, ~5 MB of RAM for the frame copy. Worth it.
# ===========================================================================

class ThreadedCamera:
    def __init__(self, cap):
        self.cap     = cap
        self.frame   = None
        self.lock    = threading.Lock()
        self.running = True
        self.thread  = threading.Thread(target=self._reader_loop, daemon=True)
        self.thread.start()
        # Wait briefly for the first frame so callers don't get None
        for _ in range(50):
            if self.frame is not None:
                break
            time.sleep(0.02)

    def _reader_loop(self):
        while self.running:
            ret, frm = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frm
            else:
                time.sleep(0.001)

    def read(self):
        """Returns (ret, frame) -- ret is False only if no frame ever arrived."""
        with self.lock:
            if self.frame is None:
                return False, None
            # Return a copy so the consumer can draw on it without the
            # background thread overwriting mid-draw.
            return True, self.frame.copy()

    def stop(self):
        self.running = False
        self.thread.join(timeout=1.0)
        self.cap.release()

# ===========================================================================
# TUNING PARAMETERS
# All values you're likely to adjust during testing are here. Grouped by
# function. Change values here -- they propagate to the rest of the code.
# ===========================================================================

# ---- Camera ----
CAMERA_INDEX = 0                   # /dev/videoN index (0 = default)
FRAME_WIDTH  = 640                 # capture resolution width in pixels
FRAME_HEIGHT = 480                 # capture resolution height in pixels
TARGET_FPS   = 60                  # requested FPS (capped by camera hardware)

# ---- Servo (pan/tilt camera mount) ----
SERVO_PAN         = 1              # servo channel for horizontal rotation
SERVO_TILT        = 2              # servo channel for vertical rotation
PAN_CENTER        = 72             # pan angle that points straight ahead
TILT_CENTER       = 25             # tilt angle that points level
PAN_MIN,  PAN_MAX  =  5, 175      # hard pan angle limits (never exceeded)
TILT_MIN, TILT_MAX =  0,  95      # hard tilt angle limits
PAN_GAIN     = 0.03                # tilt servo degrees per pixel of error
TILT_GAIN    = 0.03                # (lower = smoother, higher = faster)
MAX_STEP_DEG = 2.0                 # max servo movement per frame (prevents lurching)
DEADBAND_PX  = 5                   # ignore errors smaller than this (stops jitter)

# ---- Body Centering (TRACKING state) ----
TRACK_CENTERED_PX      = 25       # both axes must be within this many px of
                                   # frame center before advancing to CONFIRMING
MIN_TRACK_ROTATE_SPEED = 25       # minimum motor PWM that produces physical
                                   # rotation (below this, motors stall)

# ---- Color Detection ----
MIN_CONTOUR_AREA = 1500            # minimum blob area in px² (filters noise)

# ---- Blacklist (rejected target exclusion) ----
BLACKLIST_DURATION        = 5.0    # seconds a rejected target stays blocked
BLACKLIST_RADIUS_PX       = 100    # pixel radius of the exclusion zone
BLACKLIST_TRACK_RADIUS_PX = 140    # max px jump per frame for sticky tracking

# ---- Approach (driving toward confirmed target) ----
APPROACH_SPEED      = 35           # motor PWM while approaching target
ARRIVAL_DISTANCE_CM = 20.0         # sonar distance that triggers ARRIVED state
TARGET_LOCK_PX      = 80           # suppress obstacle check when target is
                                   # within this many px of frame center
ARRIVAL_MIN_AREA    = 0.08         # target must fill this fraction of frame
                                   # to confirm arrival (prevents false triggers)

# ---- Yaw PID (keeps target centered while driving forward) ----
YAW_KP             = 0.15         # proportional gain (main correction force)
YAW_KI             = 0.01         # integral gain (eliminates steady-state drift)
YAW_KD             = 0.05         # derivative gain (damps oscillation)
YAW_OUTPUT_LIMIT   = 35           # max wheel-speed bias from PID output
YAW_INTEGRAL_LIMIT = 200          # anti-windup cap on integral term
YAW_DEADBAND_PX    = 8            # ignore errors below this (prevents twitch)

# ---- Obstacle Detection ----
OBSTACLE_DISTANCE_CM = 20.0       # sonar distance that triggers obstacle scan

# ---- Obstacle Scanning (rotate body L/R, read sonar) ----
SCAN_ROTATE_ANGLE_DEG  = 90.0     # degrees to rotate each direction during scan
SCAN_SETTLE_TIME       = 0.25     # seconds to wait after rotation before reading
SCAN_SAMPLES           = 3        # sonar samples to average per side

# ---- Rotation Calibration ----
ROTATION_STARTUP_COMP_S = 0.15    # extra seconds added to the FIRST rotation
                                   # from a dead stop to compensate for motor
                                   # ramp-up lag (subsequent rotations skip this)

# ---- Wall-Following (AVOIDING state) ----
AVOID_FORWARD_DURATION = 1.0      # seconds of forward travel between wall peeks
AVOID_SPEED            = 50       # motor PWM during avoidance forward drive
WALL_FOLLOW_TOLERANCE  = 5.0      # ±cm: wall is "still there" if peek reads
                                   # within this range of the original distance
CORNER_CLEAR_SPEED     = 60       # motor PWM for end-of-wall corner strafe
CORNER_CLEAR_DURATION  = 0.3      # seconds of strafe to clear the corner edge

# ---- Search (rotating to find target) ----
SEARCH_ROTATE_SPEED = 40          # motor PWM for in-place rotation (also used
                                   # by scan, dance, and all non-PID rotations)
SEARCH_ROTATE_STEP  = 0.10        # seconds per rotation step between checks
SEARCH_MAX_STEPS    = 35          # max steps before nudging forward and retrying

# ---- Dead Reckoning (calibrated on actual surface) ----
DEG_PER_SECOND_ROTATE = 126.0     # rotation rate at SEARCH_ROTATE_SPEED
                                   # (measured: 252° in 2.0s at speed 40)
CM_PER_SECOND_FORWARD = 35.25     # forward speed at APPROACH_SPEED

# ===========================================================================
# COLOR REGISTRY
# Key  : integer 1-8 (maps to keyboard keys)
# Value: (label, box_color_BGR, hsv_lower, hsv_upper, hsv_lower2, hsv_upper2)
#
# hsv_lower2 / hsv_upper2 are only used for Red, which wraps around the
# hue boundary (0 and 180 are the same color in OpenCV's 0-180 hue scale).
# All other colors pass None for those two fields.
# ===========================================================================

COLORS = {
    1: ("Red",    (0,   0,   200),
        np.array([0,   80,  40]),  np.array([10,  255, 255]),
        np.array([170, 80,  40]),  np.array([180, 255, 255])),

    2: ("Orange", (0,   130, 255),
        np.array([10,  80,  80]),  np.array([20,  255, 255]),
        None, None),

    3: ("Yellow", (0,   215, 255),
        np.array([20,  80,  80]),  np.array([35,  255, 255]),
        None, None),

    4: ("Green",  (0,   200, 0),
        np.array([40,  80,  40]),  np.array([80,  255, 255]),
        None, None),

    5: ("Blue",   (255, 100, 0),
        np.array([100, 80,  40]),  np.array([130, 255, 255]),
        None, None),

    6: ("Purple", (200, 0,   200),
        np.array([130, 50,  40]),  np.array([160, 255, 255]),
        None, None),

    7: ("White",  (200, 200, 200),
        np.array([0,   0,   170]), np.array([180, 80,  255]),
        None, None),

    8: ("Black",  (80,  80,  80),
        np.array([0,   0,   0]),   np.array([180, 255, 50]),
        None, None),
}

# ===========================================================================
# STATES
# ===========================================================================

SEARCHING         = 'SEARCHING'
TRACKING          = 'TRACKING'
CONFIRMING        = 'CONFIRMING'
APPROACHING       = 'APPROACHING'
SCANNING_OBSTACLE = 'SCANNING_OBSTACLE'
AVOIDING          = 'AVOIDING'
ARRIVED           = 'ARRIVED'

FRAME_CX   = FRAME_WIDTH  // 2
FRAME_CY   = FRAME_HEIGHT // 2
FRAME_AREA = FRAME_WIDTH * FRAME_HEIGHT

STATE_COLORS = {
    SEARCHING:         (100, 100, 100),
    TRACKING:          (0,   200, 255),
    CONFIRMING:        (0,   255, 150),
    APPROACHING:       (0,   200, 0),
    SCANNING_OBSTACLE: (0,   140, 255),
    AVOIDING:          (0,   80,  255),
    ARRIVED:           (0,   255, 0),
}

# LED bar: 0=red 1=green 2=blue 3=yellow 4=purple 5=cyan 6=white
# None = managed by the state runner itself (ARRIVED flashes during dance).
STATE_LED_COLORS = {
    SEARCHING:         5,     # cyan
    TRACKING:          3,     # yellow
    CONFIRMING:        6,     # white
    APPROACHING:       1,     # green
    SCANNING_OBSTACLE: 3,     # yellow
    AVOIDING:          0,     # red
    ARRIVED:           None,
}


# ===========================================================================
# COLOR SELECTION SCREEN
# ===========================================================================

def draw_selection_screen(frame):
    """
    Draws a semi-transparent selection menu over the live camera frame.
    Two-column layout: color swatch + key + name for each option.
    """
    overlay = frame.copy()
    cv2.rectangle(overlay, (70, 50), (570, 430), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, 'SELECT TARGET COLOR',
                (145, 98),  font, 0.9,  (0, 255, 0),       2)
    cv2.putText(frame, 'Press a number key to begin tracking',
                (130, 128), font, 0.52, (170, 170, 170),    1)

    # Two-column grid: keys 1-4 left, 5-8 right
    for key in range(1, 9):
        label, box_color, _, _, _, _ = COLORS[key]
        col = (key - 1) % 2
        row = (key - 1) // 2
        x = 120 + col * 250
        y = 185 + row * 58

        # Color swatch
        cv2.rectangle(frame, (x, y - 22), (x + 26, y + 4), box_color, -1)
        cv2.rectangle(frame, (x, y - 22), (x + 26, y + 4), (200, 200, 200), 1)

        cv2.putText(frame, f'[{key}]  {label}',
                    (x + 34, y), font, 0.72, (220, 220, 220), 2)

    cv2.putText(frame, 'Q: quit',
                (280, 415), font, 0.5, (110, 110, 110), 1)
    return frame


def run_color_selection(camera):
    """
    Runs the on-screen color selection loop.
    Blocks until the user presses 1-8 or Q.
    Returns the selected key (int) or None if quit.
    Accepts either a cv2.VideoCapture or a ThreadedCamera (both expose .read()).
    """
    print('Color selection active -- press 1-8 on the camera window.')
    while True:
        ret, frame = camera.read()
        if not ret:
            time.sleep(0.01)
            continue
        frame = draw_selection_screen(frame)
        cv2.imshow('Color Block Tracker', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            return None
        for k in range(1, 9):
            if key == ord(str(k)):
                print(f'Selected color: {COLORS[k][0]}')
                return k


# ===========================================================================
# DETECTION
# ===========================================================================

def find_color_block(frame, color_key, blacklist):
    """
    Find the largest blob of the selected color, excluding blacklisted zones.

    Detection pipeline:
      1. Convert frame to HSV (separates color from brightness).
      2. Threshold with inRange to create a binary mask.
      3. For red: OR two masks together (handles hue wrap-around).
      4. Morphological close+open: fill holes, remove speckle noise.
      5. Find external contours; filter by MIN_CONTOUR_AREA.
      6. STICKY BLACKLIST UPDATE -- snap each live blacklist entry to the
         nearest contour center within BLACKLIST_TRACK_RADIUS_PX, so the
         entry "rides" the rejected blob across the frame as the bot rotates.
      7. Filter out contours whose centers fall within BLACKLIST_RADIUS_PX
         of any (now-updated) entry.
      8. Return the largest surviving contour as a target dict.

    WHY STICKY BLACKLIST?
      A static pixel-coordinate blacklist breaks the moment the bot rotates:
      the rejected object slides to a new pixel position and escapes the
      exclusion zone. Tracking the entry to the actual blob each frame
      keeps the exclusion attached to the physical object.

    WHY NO SHAPE FILTER?
      Relying solely on color and size is more permissive -- it handles
      irregular, partially-occluded, or non-rectangular targets. The Y/N
      confirmation step is the quality gate that replaces shape filtering.

    Args:
        frame      -- BGR image (not modified)
        color_key  -- 1-8, indexes into COLORS
        blacklist  -- list of [cx, cy, expiry_time] from tracker. MUTATED in
                      place: live entries are snapped to current contour
                      positions. Must be a list of lists, not tuples.

    Returns:
        dict with keys x, y, w, h, cx, cy, offset_x, offset_y, area,
        area_ratio -- or None if nothing found.
    """
    _, _, lo, hi, lo2, hi2 = COLORS[color_key]

    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lo, hi)

    if lo2 is not None and hi2 is not None:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo2, hi2))

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    # Size filter -- one pass, also caches bounding box and center per contour
    candidates = []
    for c in contours:
        if cv2.contourArea(c) < MIN_CONTOUR_AREA:
            continue
        x, y, w, h = cv2.boundingRect(c)
        cx, cy = x + w // 2, y + h // 2
        candidates.append({'contour': c, 'x': x, 'y': y, 'w': w, 'h': h,
                           'cx': cx, 'cy': cy,
                           'area': cv2.contourArea(c)})

    # --- STICKY BLACKLIST UPDATE ---
    # For each live entry, snap to the nearest candidate center within the
    # tracking radius. This must happen BEFORE filtering, because we want
    # entries to track even contours they would later exclude.
    now = time.time()
    for entry in blacklist:
        if entry[2] <= now:
            continue   # expired
        bx, by = entry[0], entry[1]
        best_d   = BLACKLIST_TRACK_RADIUS_PX
        best_pos = None
        for cand in candidates:
            d = math.hypot(cand['cx'] - bx, cand['cy'] - by)
            if d < best_d:
                best_d   = d
                best_pos = (cand['cx'], cand['cy'])
        if best_pos is not None:
            entry[0] = best_pos[0]
            entry[1] = best_pos[1]

    # --- BLACKLIST EXCLUSION FILTER ---
    active = [(e[0], e[1]) for e in blacklist if e[2] > now]
    valid  = [cand for cand in candidates
              if not any(math.hypot(cand['cx'] - bx, cand['cy'] - by)
                         < BLACKLIST_RADIUS_PX
                         for bx, by in active)]

    if not valid:
        return None

    best = max(valid, key=lambda c: c['area'])
    return {
        'x': best['x'], 'y': best['y'], 'w': best['w'], 'h': best['h'],
        'cx': best['cx'], 'cy': best['cy'],
        'offset_x':   best['cx'] - FRAME_CX,
        'offset_y':   best['cy'] - FRAME_CY,
        'area':       best['area'],
        'area_ratio': best['area'] / FRAME_AREA,
    }


# ===========================================================================
# PID CONTROLLER
# Generic single-input single-output PID with anti-windup and output clamping.
# Used for yaw correction during APPROACHING -- the bot rotates while driving
# forward to keep the target horizontally centered in frame.
# ===========================================================================

class PIDController:
    """
    Standard discrete PID:
        u(t) = Kp*e + Ki*integral(e dt) + Kd*(de/dt)

    HOW IT WORKS HERE:
      error  -- target.offset_x in pixels (positive = target right of center)
      output -- wheel-speed bias in motor units, clamped to +/- output_limit
      Apply  -- LEFT wheels  += output    RIGHT wheels  -= output
                A positive bias yaws the bot clockwise toward a right-side target.

    ANTI-WINDUP:
      Without this, the integral term keeps growing while the bot is saturated
      (e.g., target way off-screen) and overshoots wildly when it finally catches
      up. We cap the integral at +/- integral_limit so it can never accumulate
      more authority than we want.

    DEADBAND:
      Tiny pixel errors are ignored. This prevents jitter when the target is
      already nearly centered -- the bot drives perfectly straight instead of
      micro-correcting on every frame.
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
        """Call once per frame with the current error. Returns the control output."""
        now = time.time()
        if self._last_time is None:
            dt = 0.0
        else:
            dt = now - self._last_time
        self._last_time = now

        # Deadband -- treat small errors as zero
        if abs(error) < self.deadband:
            error = 0.0

        # Integral with anti-windup clamp
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


# ===========================================================================
# DEAD RECKONING
# ===========================================================================

class DeadReckoning:
    """
    Estimates robot position and heading relative to last block sighting.

    Coordinate system (origin = where block was last seen):
      heading : degrees, 0 = facing block, positive = rotated clockwise
      x       : cm, positive = right
      y       : cm, positive = forward

    After an avoidance maneuver, estimated_block_bearing gives the rotation
    angle needed to face back toward the block's last known position.

    WHY DEAD RECKONING?
      The camera can't see the block during avoidance. Dead reckoning
      accumulates all robot movements (speed x time) to estimate the current
      position. It drifts over long distances but is accurate enough for the
      short avoidance maneuvers here. Adding wheel encoders or an IMU later
      would replace the time-based estimates with measured values.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.heading = 0.0
        self.x       = 0.0
        self.y       = 0.0

    def rotate(self, direction, duration):
        """direction: +1 = clockwise, -1 = counterclockwise"""
        deg = DEG_PER_SECOND_ROTATE * duration * direction
        self.heading += deg
        dbg(f'[DR] Rotated {deg:+.1f}deg  heading now {self.heading:+.1f}deg')

    def move_forward(self, duration):
        dist = CM_PER_SECOND_FORWARD * duration
        rad  = math.radians(self.heading)
        self.x += dist * math.sin(rad)
        self.y += dist * math.cos(rad)
        dbg(f'[DR] Forward {dist:.1f}cm  pos ({self.x:.1f}, {self.y:.1f})')


# ===========================================================================
# TRACKER STATE MACHINE
# ===========================================================================

class ColorBlockTracker:

    def __init__(self, robot, color_key):
        self.robot      = robot
        self.color_key  = color_key
        self.color_name = COLORS[color_key][0]
        self.box_color  = COLORS[color_key][1]

        self.state            = SEARCHING
        self.dr               = DeadReckoning()
        self.pan              = PAN_CENTER
        self.tilt             = TILT_CENTER
        self.search_steps     = 0
        self.search_direction = +1   # +1 = clockwise, -1 = CCW.
                                     # Set to -avoid_direction after obstacle
                                     # avoidance so we search toward the target.
        self.avoid_phase      = 0
        self.avoid_direction  = +1   # +1 = rotate right, -1 = rotate left
        self.wall_ref_dist   = 0.0  # original obstacle distance for wall-peek comparison
        self.last_action_time = time.time()

        # Yaw correction PID for APPROACHING (keeps target horizontally centered)
        self.yaw_pid = PIDController(
            kp=YAW_KP, ki=YAW_KI, kd=YAW_KD,
            output_limit=YAW_OUTPUT_LIMIT,
            integral_limit=YAW_INTEGRAL_LIMIT,
            deadband=YAW_DEADBAND_PX,
        )

        # Obstacle scanning state (phased / non-blocking)
        self.scan_phase       = 0
        self.scan_phase_start = 0.0
        self.scan_left_dist   = 999.0
        self.scan_right_dist  = 999.0
        self.is_initial_scan  = True   # True for first scan from APPROACHING,
                                       # False for re-scans during wall-following

        # Arrival dance state
        self.dance_step          = 0
        self.dance_step_start    = 0.0
        self.dance_led_last      = 0.0
        self.dance_led_on        = False
        self.dance_complete      = False

        # Each entry: (cx, cy, expiry_time)
        # Contours near a live entry are skipped in find_color_block()
        self.blacklist = []

        self.robot.Ctrl_Ulatist_Switch(1)   # Power on ultrasonic sensor
        # The Yahboom sonar takes a few hundred ms after enable before reads
        # return real values. Without this delay the first SCANNING_OBSTACLE
        # cycle gets all-zero reads and the bot loops trapped/escape forever.
        time.sleep(0.5)
        # Sanity-check: try a few reads and warn if they're all bogus.
        warmup = [self._read_sonar() for _ in range(3)]
        if all(r >= 999.0 for r in warmup):
            print('[SONAR] WARNING: all warmup reads failed -- '
                  'sensor may not be powered/connected. '
                  'Run ultrasonic_test.py to verify.')
        else:
            print(f'[SONAR] Warmup reads: {[round(r,1) for r in warmup]} cm')
        self._center_camera()
        self._set_led_for_state(self.state)

    def _center_camera(self):
        self.robot.Ctrl_Servo(SERVO_PAN,  PAN_CENTER)
        self.robot.Ctrl_Servo(SERVO_TILT, TILT_CENTER)
        self.pan  = PAN_CENTER
        self.tilt = TILT_CENTER

    def _drive(self, fl, fr, rl, rr):
        """
        Drive all four Mecanum wheels individually.
        Motor index mapping (confirmed from Yahboom library):
          0 = front-left   1 = front-right
          2 = rear-left    3 = rear-right
        Positive = forward, negative = backward for each wheel.

        All 4 commands are sent every call — no caching. The Yahboom I2C
        bus silently drops writes occasionally (bare except in write_array),
        and the old cache meant a dropped write was never retried. At 4
        writes × ~1ms each = ~4ms per call, the cost is acceptable for the
        reliability gain.
        """
        self.robot.Ctrl_Muto(0, fl)
        self.robot.Ctrl_Muto(1, fr)
        self.robot.Ctrl_Muto(2, rl)
        self.robot.Ctrl_Muto(3, rr)

    def _stop(self):
        """Stop all four wheels."""
        for i in range(4):
            self.robot.Ctrl_Muto(i, 0)

    def _read_sonar(self):
        """
        Read ultrasonic distance in cm.
        The sensor stores the result across two I2C registers:
          0x1b = high byte,  0x1a = low byte
        Combined: (high << 8 | low) / 10.0 gives distance in cm.
        Returns 999.0 on read failure so the bot treats the path as clear.

        IMPORTANT: A successful read of 0.0 is treated as a FAILED read,
        not as "obstacle touching the sensor." In practice 0.0 only happens
        when the sensor is powered down, mid-warmup, or the I2C transaction
        silently returned zero bytes. Treating 0.0 as a real "0 cm distance"
        used to cause the bot to declare itself trapped on every scan and
        loop endlessly through SCANNING_OBSTACLE.
        """
        try:
            dist_H = self.robot.read_data_array(0x1b, 1)[0]
            dist_L = self.robot.read_data_array(0x1a, 1)[0]
            raw    = (dist_H << 8 | dist_L) / 10.0
        except Exception as e:
            dbg(f'[SONAR] Read error: {e}')
            return 999.0

        if raw <= 0.0:
            return 999.0
        return raw

    def _read_sonar_avg(self, samples=SCAN_SAMPLES):
        """
        Average several sonar reads to reject single-frame noise.
        Discards 999.0 (failed) reads from the average; if every sample is a
        failed read, returns 999.0 so the caller sees "clear / unknown."
        """
        valid = []
        for _ in range(samples):
            r = self._read_sonar()
            if r < 999.0:
                valid.append(r)
            time.sleep(0.02)
        if not valid:
            return 999.0
        return sum(valid) / len(valid)

    def _start_rotation(self, direction):
        """
        Begin an in-place rotation at SEARCH_ROTATE_SPEED. Returns immediately.
        Use _rotation_duration_for(degrees) to compute how long to wait, then
        check elapsed time in the calling state runner before stopping.
        """
        speed = SEARCH_ROTATE_SPEED
        self._drive( speed * direction,  speed * direction,
                    -speed * direction, -speed * direction)

    @staticmethod
    def _rotation_duration_for(degrees, from_stop=False):
        """Seconds at SEARCH_ROTATE_SPEED to rotate the given angle.
        from_stop=True adds startup compensation for motor ramp-up lag —
        only needed on the first rotation from a true dead stop. Subsequent
        rotations (motors still warm, residual momentum) don't need it."""
        base = degrees / DEG_PER_SECOND_ROTATE
        return base + ROTATION_STARTUP_COMP_S if from_stop else base

    def _set_led_for_state(self, state):
        """
        Apply the LED bar color for the given state.
        ARRIVED is excluded -- the dance manages its own flashing.
        """
        color_code = STATE_LED_COLORS.get(state)
        if color_code is None:
            return
        try:
            self.robot.Ctrl_WQ2812_ALL(1, color_code)
        except Exception as e:
            dbg(f'[LED] error setting color: {e}')

    def _leds_off(self):
        try:
            self.robot.Ctrl_WQ2812_ALL(0, 0)
        except Exception:
            pass

    def _set_state(self, new_state):
        if new_state != self.state:
            print(f'[STATE] {self.state} -> {new_state}')
            self.state            = new_state
            self.last_action_time = time.time()
            self._set_led_for_state(new_state)
            # Reset PID whenever we (re)enter any state that uses body
            # rotation feedback so old integral/derivative state from a
            # previous run doesn't kick the bot sideways on the first frame.
            if new_state in (TRACKING, APPROACHING):
                self.yaw_pid.reset()

    def _elapsed(self):
        return time.time() - self.last_action_time

    def _update_servos(self, target):
        """
        Tilt-only servo controller -- pan is locked at center.

        WHY LOCK PAN?
          If pan tracks horizontally, the robot drives toward wherever the
          camera is pointing -- potentially diagonally away from the target.
          Locking pan at center means the robot always drives straight ahead,
          directly toward the object. Tilt still floats so the camera stays
          on the target vertically as the bot approaches and the object rises
          in frame.
        """
        # Pan is intentionally not updated -- locked at PAN_CENTER
        if abs(target['offset_y']) > DEADBAND_PX:
            raw  = target['offset_y'] * TILT_GAIN
            step = float(np.clip(raw, -MAX_STEP_DEG, MAX_STEP_DEG))
            self.tilt = int(np.clip(self.tilt - step, TILT_MIN, TILT_MAX))
            self.robot.Ctrl_Servo(SERVO_TILT, self.tilt)

    def _is_centered(self, target):
        """
        Body is considered centered on the target when both horizontal and
        vertical pixel offsets are within TRACK_CENTERED_PX. Horizontal
        centering is achieved by body rotation in _run_tracking; vertical
        centering is achieved by camera tilt in _update_servos.
        """
        return (abs(target['offset_x']) <= TRACK_CENTERED_PX and
                abs(target['offset_y']) <= TRACK_CENTERED_PX)

    def _prune_blacklist(self):
        """Remove expired blacklist entries at the start of each frame."""
        now = time.time()
        self.blacklist = [e for e in self.blacklist if e[2] > now]

    # --- Public confirmation API (called from main() on keypress) ----------

    def confirm_target(self):
        """User pressed Y. Advance from CONFIRMING to APPROACHING."""
        if self.state == CONFIRMING:
            self._set_state(APPROACHING)

    def reject_target(self, target):
        """
        User pressed N. Blacklist the target's screen region, then rescan.

        The blacklist entry uses the target's frame-center coordinates (cx, cy).
        Any future contour within BLACKLIST_RADIUS_PX of this point is skipped
        for BLACKLIST_DURATION seconds. This prevents the bot from immediately
        re-locking the same rejected object.
        """
        if self.state == CONFIRMING:
            expiry = time.time() + BLACKLIST_DURATION
            # Use a list (not tuple) so find_color_block can update the
            # entry's coordinates as the rejected blob slides across the frame.
            self.blacklist.append([target['cx'], target['cy'], expiry])
            print(f'[BLACKLIST] ({target["cx"]}, {target["cy"]}) '
                  f'blocked for {BLACKLIST_DURATION:.0f}s (sticky)')
            self._set_state(SEARCHING)

    # -----------------------------------------------------------------------
    # SEARCHING
    # Rotate bot in self.search_direction increments.
    # Stops and transitions to TRACKING the moment a blob is detected.
    # After a full rotation, nudges forward and restarts scan.
    #
    # search_direction is +1 (clockwise) by default, but set to
    # -avoid_direction after obstacle avoidance so the bot searches toward
    # the target instead of away from it.
    # -----------------------------------------------------------------------

    def _run_searching(self, target):
        if target:
            self._stop()
            self.search_steps     = 0
            self.search_direction = +1   # reset to default for next search
            self.dr.reset()
            self._set_state(TRACKING)
            return

        if self._elapsed() >= SEARCH_ROTATE_STEP:
            sd = self.search_direction
            self._drive( SEARCH_ROTATE_SPEED * sd,  SEARCH_ROTATE_SPEED * sd,
                        -SEARCH_ROTATE_SPEED * sd, -SEARCH_ROTATE_SPEED * sd)
            self.dr.rotate(sd, SEARCH_ROTATE_STEP)
            self.search_steps    += 1
            self.last_action_time = time.time()

            if self.search_steps >= SEARCH_MAX_STEPS:
                print('[SEARCH] Full rotation complete, advancing forward.')
                self._stop()
                self._drive(APPROACH_SPEED, APPROACH_SPEED,
                            APPROACH_SPEED, APPROACH_SPEED)
                time.sleep(0.5)
                self.dr.move_forward(0.5)
                self._stop()
                self.search_steps = 0
                self.dr.reset()

    # -----------------------------------------------------------------------
    # TRACKING
    # Target visible. Rotate body horizontally and tilt camera vertically to
    # center the target. Once both axes are within TRACK_CENTERED_PX, advance
    # to CONFIRMING.
    #
    # Body rotation uses the yaw PID (same one used by APPROACHING). It is
    # reset on entry to TRACKING so old state from a previous session can't
    # kick the bot. The PID output is treated as a wheel rotation speed and
    # bumped to MIN_TRACK_ROTATE_SPEED if non-zero but below the motor
    # deadzone -- otherwise tiny corrections produce zero physical motion
    # and the bot would sit forever just shy of centered.
    # -----------------------------------------------------------------------

    def _run_tracking(self, target):
        if not target:
            self._stop()
            self._set_state(SEARCHING)
            return

        # Vertical centering via camera tilt
        self._update_servos(target)

        # Horizontal centering via body rotation
        x_off = target['offset_x']
        if abs(x_off) > TRACK_CENTERED_PX:
            raw_speed = self.yaw_pid.update(float(x_off))
            speed     = int(raw_speed)

            # Apply motor-deadzone floor: if PID asked for any nonzero motion
            # but the magnitude is below what the motors can actually act on,
            # bump it up to the floor (preserving sign).
            if 0 < abs(speed) < MIN_TRACK_ROTATE_SPEED:
                speed = MIN_TRACK_ROTATE_SPEED if speed > 0 else -MIN_TRACK_ROTATE_SPEED

            # In-place rotation: left wheels +speed, right wheels -speed.
            # Positive speed = clockwise = target was to the right.
            self._drive( speed,  speed, -speed, -speed)
        else:
            self._stop()

        # Advance to CONFIRMING only when BOTH axes are inside the deadband
        if self._is_centered(target):
            self._stop()
            self._set_state(CONFIRMING)

    # -----------------------------------------------------------------------
    # CONFIRMING
    # Bot is fully stopped. Camera is centered on target.
    # This state is passive -- it does not auto-advance.
    # main() handles Y/N keypresses and calls confirm_target()/reject_target().
    # If the target disappears before the user responds, drop back to SEARCHING.
    # -----------------------------------------------------------------------

    def _run_confirming(self, target):
        if not target:
            print('[CONFIRM] Target lost before confirmation -- returning to scan.')
            self._set_state(SEARCHING)

    # -----------------------------------------------------------------------
    # APPROACHING
    # Confirmed target. Drive forward while yaw-correcting (PID) to keep the
    # target horizontally centered. Stop on:
    #   - obstacle within OBSTACLE_DISTANCE_CM  -> SCANNING_OBSTACLE
    #   - sonar reading <= ARRIVAL_DISTANCE_CM  -> ARRIVED
    # -----------------------------------------------------------------------

    def _run_approaching(self, target):
        if not target:
            self._stop()
            self._set_state(SEARCHING)
            return

        # Tilt-only servo update so the camera stays on the target vertically
        self._update_servos(target)

        dist = self._read_sonar()

        # Arrival check FIRST -- but require visual confirmation that the
        # target is actually large in frame. Sonar alone can't tell whether
        # the close object is the target or a stray obstacle that wandered
        # into the cone. If sonar says "close" but the target is still small
        # in frame, the close thing is something else -- treat it as an
        # obstacle and let the scan logic decide.
        if dist <= ARRIVAL_DISTANCE_CM:
            if target['area_ratio'] >= ARRIVAL_MIN_AREA:
                self._stop()
                print(f'[ARRIVED] Sonar {dist:.1f}cm  area '
                      f'{target["area_ratio"]*100:.1f}%')
                self._set_state(ARRIVED)
                return
            else:
                # Sonar sees something close, but it's not our target.
                # Fall through to obstacle handling below.
                dbg(f'[REJECT-ARRIVAL] Sonar {dist:.1f}cm but target '
                    f'only {target["area_ratio"]*100:.1f}% of frame')

        # Obstacle check -- something in the way that is NOT the target.
        # Suppressed when the target is currently locked near frame-center
        # AND large enough in frame that sonar is plausibly seeing it.
        # If the target is centered but still small, sonar must be seeing
        # something else, so we still want to scan.
        target_locked = (abs(target['offset_x']) <= TARGET_LOCK_PX and
                         target['area_ratio'] >= ARRIVAL_MIN_AREA)
        if dist < OBSTACLE_DISTANCE_CM and not target_locked:
            self._stop()
            print(f'[OBSTACLE] {dist:.1f}cm -- scanning sides')
            self.scan_phase       = 0
            self.scan_phase_start = time.time()
            self.wall_ref_dist    = dist
            self.is_initial_scan  = True
            self._set_state(SCANNING_OBSTACLE)
            return

        # ---- Yaw correction via PID ----
        # error = pixels the target is off-center horizontally
        # output = wheel-speed bias (positive = rotate right toward target)
        error = float(target['offset_x'])
        bias  = self.yaw_pid.update(error)

        left  = int(APPROACH_SPEED + bias)
        right = int(APPROACH_SPEED - bias)
        self._drive(left, left, right, right)

    # -----------------------------------------------------------------------
    # SCANNING_OBSTACLE  (non-blocking, rotation-optimized)
    #
    # Scans left then right. After the 180° swing to the right, the bot is
    # already facing 90° RIGHT of its original heading. Instead of rotating
    # back to center and then turning again, we exploit the current heading:
    #
    #   Phase 0: start 90° LEFT rotation
    #   Phase 1: wait + settle
    #   Phase 2: read LEFT sonar, start 180° RIGHT rotation
    #   Phase 3: wait + settle
    #   Phase 4: read RIGHT sonar, DECIDE:
    #       RIGHT wins → already facing right, enter AVOIDING drive loop
    #       LEFT wins  → start 180° flip left → Phase 5
    #       TRAPPED    → start 90° more right (→ backward) → Phase 7
    #   Phase 5: wait for 180° flip → enter AVOIDING drive loop
    #
    # Saves a full 90° rotation (~0.7s) vs the old return-to-center approach.
    # -----------------------------------------------------------------------

    def _run_scanning_obstacle(self, target):
        elapsed = time.time() - self.scan_phase_start

        if self.scan_phase == 0:
            self._start_rotation(-1)   # rotate LEFT
            self.scan_phase       = 1
            self.scan_phase_start = time.time()

        elif self.scan_phase == 1:
            dur = self._rotation_duration_for(SCAN_ROTATE_ANGLE_DEG, from_stop=True)
            if elapsed >= dur:
                self._stop()
                self.dr.rotate(-1, dur)
                self.scan_phase       = 2
                self.scan_phase_start = time.time()
            else:
                self._start_rotation(-1)   # re-send each frame

        elif self.scan_phase == 2:
            if elapsed >= SCAN_SETTLE_TIME:
                self.scan_left_dist = self._read_sonar_avg()
                dbg(f'[SCAN] Left: {self.scan_left_dist:.1f}cm')
                self._start_rotation(+1)    # swing 180° RIGHT
                self.scan_phase       = 3
                self.scan_phase_start = time.time()

        elif self.scan_phase == 3:
            if elapsed >= self._rotation_duration_for(2 * SCAN_ROTATE_ANGLE_DEG):
                self._stop()
                self.dr.rotate(+1, self._rotation_duration_for(2 * SCAN_ROTATE_ANGLE_DEG))
                self.scan_phase       = 4
                self.scan_phase_start = time.time()
            else:
                self._start_rotation(+1)   # re-send each frame

        elif self.scan_phase == 4:
            if elapsed >= SCAN_SETTLE_TIME:
                self.scan_right_dist = self._read_sonar_avg()
                dbg(f'[SCAN] Right: {self.scan_right_dist:.1f}cm')
                self._scan_decide()

        elif self.scan_phase == 5:
            # LEFT won: waiting for 180° flip to face left
            if elapsed >= self._rotation_duration_for(2 * SCAN_ROTATE_ANGLE_DEG):
                self._stop()
                self.dr.rotate(-1, self._rotation_duration_for(2 * SCAN_ROTATE_ANGLE_DEG))
                self._enter_avoiding_drive()
            else:
                self._start_rotation(-1)   # re-send each frame

    def _scan_decide(self):
        """
        Called at end of phase 4. Bot faces 90° RIGHT of original heading.

        Decision priority:
          1. If this is a RE-SCAN (during wall-following), prefer the direction
             back toward the target (-avoid_direction) if that side has enough
             clearance (> OBSTACLE_DISTANCE_CM).
          2. If the preferred side is blocked, go the other way.
          3. If both sides are tight, go whichever has more clearance (best
             effort — no trapped/escape logic).
          4. If this is the INITIAL scan (from APPROACHING), there's no prior
             avoid_direction, so just pick the clearer side.
        """
        left  = self.scan_left_dist
        right = self.scan_right_dist

        if self.is_initial_scan:
            # First scan from APPROACHING — no directional preference yet.
            # Pick the side with more clearance.
            chosen = self._pick_clearer_side(left, right)
        else:
            # Re-scan during wall-following. Prefer getting back toward the
            # target, which is in the -avoid_direction.
            preferred = -self.avoid_direction   # +1=right, -1=left
            chosen    = self._pick_with_preference(left, right, preferred)

        if chosen == +1:
            # RIGHT. Already facing right → drive immediately.
            self.avoid_direction = +1
            self._enter_avoiding_drive()
        else:
            # LEFT. Need 180° flip from current right-facing heading.
            self.avoid_direction = -1
            print(f'[SCAN] L={left:.0f}cm R={right:.0f}cm  -> LEFT (flipping)')
            self._start_rotation(-1)
            self.scan_phase       = 5
            self.scan_phase_start = time.time()

    def _pick_clearer_side(self, left, right):
        """No preference — just pick the side with more clearance. Tie = right."""
        if left > right + 5.0:
            print(f'[SCAN] L={left:.0f}cm R={right:.0f}cm  -> LEFT (clearer)')
            return -1
        else:
            tag = 'RIGHT' if abs(left - right) >= 5.0 else 'RIGHT (tie)'
            print(f'[SCAN] L={left:.0f}cm R={right:.0f}cm  -> {tag}')
            return +1

    def _pick_with_preference(self, left, right, preferred):
        """
        Prefer the direction back toward the target, but only if that side
        has enough clearance. "Enough" = reading > OBSTACLE_DISTANCE_CM.
        """
        pref_dist  = left if preferred == -1 else right
        other_dist = right if preferred == -1 else left
        pref_name  = 'LEFT' if preferred == -1 else 'RIGHT'
        other_name = 'RIGHT' if preferred == -1 else 'LEFT'

        if pref_dist > OBSTACLE_DISTANCE_CM:
            print(f'[SCAN] L={left:.0f}cm R={right:.0f}cm  '
                  f'-> {pref_name} (toward target, {pref_dist:.0f}cm clear)')
            return preferred
        elif other_dist > OBSTACLE_DISTANCE_CM:
            print(f'[SCAN] L={left:.0f}cm R={right:.0f}cm  '
                  f'-> {other_name} (target side blocked, detour)')
            return -preferred
        else:
            # Both tight. Best effort — go whichever has more clearance.
            if pref_dist >= other_dist:
                print(f'[SCAN] L={left:.0f}cm R={right:.0f}cm  '
                      f'-> {pref_name} (both tight, prefer target side)')
                return preferred
            else:
                print(f'[SCAN] L={left:.0f}cm R={right:.0f}cm  '
                      f'-> {other_name} (both tight, more clearance)')
                return -preferred

    def _enter_avoiding_drive(self):
        """Enter AVOIDING at phase 2 (forward drive loop), skipping the
        initial rotation since the scan already left us facing the right way."""
        self.avoid_phase      = 2
        self.last_action_time = time.time()
        self._set_state(AVOIDING)

    # -----------------------------------------------------------------------
    # AVOIDING  (non-blocking, phased: wall-following with peek-and-clear)
    #
    # Navigates around obstacles as if the bot is in a maze. After the scan
    # picks a clearer side, the bot:
    #   1. Rotates 90° toward that side.
    #   2. Drives forward for AVOID_FORWARD_DURATION.
    #   3. Stops and peeks 90° back toward the wall.
    #   4. If the wall is still there (sonar within ±WALL_FOLLOW_TOLERANCE of
    #      original distance), rotates back and repeats from step 2.
    #   5. If the wall has ended (sonar reads much farther), does a brief
    #      strafe toward the wall side to clear the corner, then SEARCHING.
    #   6. If sonar reads much CLOSER than the reference, a new perpendicular
    #      wall is closing in — go back to SCANNING_OBSTACLE to re-evaluate.
    #   7. During forward segments, sonar checks every frame for new obstacles
    #      ahead. If one appears, abort and go to SCANNING_OBSTACLE.
    #
    # Phases:
    #   0  start 90° rotation toward clearer side
    #   1  wait for rotation
    #   2  drive forward (sonar checking ahead each frame)
    #   3  start 90° peek rotation toward wall
    #   4  wait for peek rotation
    #   5  settle + read sonar, decide
    #   6  start 90° return rotation (back to travel direction)
    #   7  wait for return rotation → loop to phase 2
    #   8  start corner-clearing strafe
    #   9  wait for strafe → SEARCHING
    # -----------------------------------------------------------------------

    def _run_avoiding(self, target):
        elapsed = self._elapsed()
        d = self.avoid_direction       # +1 = right (CW), -1 = left (CCW)
        peek_dir = -d                  # peek toward the wall (opposite side)

        # --- Phase 0: start initial 90° rotation toward clearer side ---
        if self.avoid_phase == 0:
            self._start_rotation(d)
            self.avoid_phase      = 1
            self.last_action_time = time.time()

        # --- Phase 1: wait for initial rotation ---
        elif self.avoid_phase == 1:
            dur = self._rotation_duration_for(SCAN_ROTATE_ANGLE_DEG, from_stop=True)
            if elapsed >= dur:
                self._stop()
                self.dr.rotate(d, dur)
                print(f'[AVOID] Rotated {SCAN_ROTATE_ANGLE_DEG:.0f}° '
                      f'{"RIGHT" if d > 0 else "LEFT"} -- driving forward')
                self.avoid_phase      = 2
                self.last_action_time = time.time()
            else:
                self._start_rotation(d)   # re-send each frame

        # --- Phase 2: drive forward, sonar checking ahead ---
        elif self.avoid_phase == 2:
            dist = self._read_sonar()
            if dist < OBSTACLE_DISTANCE_CM and dist > 0:
                print(f'[AVOID] New obstacle ahead at {dist:.1f}cm '
                      f'-- re-scanning')
                self._stop()
                self.scan_phase       = 0
                self.scan_phase_start = time.time()
                self.wall_ref_dist    = dist
                self.is_initial_scan  = False
                self._set_state(SCANNING_OBSTACLE)
                return

            if elapsed < AVOID_FORWARD_DURATION:
                self._drive(AVOID_SPEED, AVOID_SPEED,
                            AVOID_SPEED, AVOID_SPEED)
            else:
                self._stop()
                self.dr.move_forward(AVOID_FORWARD_DURATION)
                dbg(f'[AVOID] Forward segment done -- peeking toward wall')
                self.avoid_phase      = 3
                self.last_action_time = time.time()

        # --- Phase 3: start peek rotation toward wall ---
        elif self.avoid_phase == 3:
            self._start_rotation(peek_dir)
            self.avoid_phase      = 4
            self.last_action_time = time.time()

        # --- Phase 4: wait for peek rotation ---
        elif self.avoid_phase == 4:
            if elapsed >= self._rotation_duration_for(SCAN_ROTATE_ANGLE_DEG):
                self._stop()
                self.dr.rotate(peek_dir,
                               self._rotation_duration_for(SCAN_ROTATE_ANGLE_DEG))
                self.avoid_phase      = 5
                self.last_action_time = time.time()
            else:
                self._start_rotation(peek_dir)   # re-send each frame

        # --- Phase 5: settle + read sonar, decide ---
        elif self.avoid_phase == 5:
            if elapsed >= SCAN_SETTLE_TIME:
                peek_dist = self._read_sonar_avg()
                ref       = self.wall_ref_dist
                lo        = ref - WALL_FOLLOW_TOLERANCE
                hi        = ref + WALL_FOLLOW_TOLERANCE

                if lo <= peek_dist <= hi:
                    # Wall still there — rotate back and keep going
                    print(f'[AVOID] Wall still there (peek={peek_dist:.0f}cm, '
                          f'ref={ref:.0f}±{WALL_FOLLOW_TOLERANCE:.0f}) '
                          f'-- continuing forward')
                    self.avoid_phase      = 6
                    self.last_action_time = time.time()

                elif peek_dist > hi:
                    # Wall ended — bot is already facing toward the target
                    # (from the peek rotation). Just strafe in the travel
                    # direction to clear the corner edge, then SEARCHING.
                    print(f'[AVOID] Wall ended (peek={peek_dist:.0f}cm, '
                          f'ref was {ref:.0f}cm) -- clearing corner')
                    self.avoid_phase      = 10
                    self.last_action_time = time.time()

                else:
                    # Reading much closer than expected — new wall closing in
                    # (maze corner). Re-scan from this position.
                    print(f'[AVOID] New wall detected (peek={peek_dist:.0f}cm, '
                          f'ref was {ref:.0f}cm) -- re-scanning')
                    self._stop()
                    self.scan_phase       = 0
                    self.scan_phase_start = time.time()
                    self.wall_ref_dist    = peek_dist
                    self.is_initial_scan  = False
                    self._set_state(SCANNING_OBSTACLE)
                    return

        # --- Phase 6: start rotation back to travel direction ---
        elif self.avoid_phase == 6:
            self._start_rotation(d)
            self.avoid_phase      = 7
            self.last_action_time = time.time()

        # --- Phase 7: wait for return rotation → loop to forward ---
        elif self.avoid_phase == 7:
            if elapsed >= self._rotation_duration_for(SCAN_ROTATE_ANGLE_DEG):
                self._stop()
                self.dr.rotate(d, self._rotation_duration_for(SCAN_ROTATE_ANGLE_DEG))
                self.avoid_phase      = 2
                self.last_action_time = time.time()
            else:
                self._start_rotation(d)   # re-send each frame

        # --- Phase 10: strafe in travel direction to clear the corner ---
        # After the peek found the wall gone, the bot is facing toward the
        # target (peek_dir). A brief strafe in the travel direction (d) moves
        # the bot past the corner edge so it doesn't clip when approaching.
        elif self.avoid_phase == 10:
            if elapsed < CORNER_CLEAR_DURATION:
                s = CORNER_CLEAR_SPEED
                self._drive( s * d, -s * d,
                            -s * d,  s * d)
            else:
                self._stop()
                self.dr.reset()
                self.search_direction = peek_dir  # search toward target
                print('[AVOID] Corner cleared -- returning to SEARCHING')
                self._set_state(SEARCHING)

    # -----------------------------------------------------------------------
    # ARRIVED
    # Terminal state. Runs a non-blocking "swivel dance":
    #   3 sets of [+20deg right, -20deg left back through center, return to 0].
    # During the dance the LED bar flashes green slowly (~1Hz) so it is clearly
    # visible without being a strobe hazard. After the dance, LEDs go off and
    # the bot stays put until the user presses Q.
    #
    # Implementation notes:
    #   - Steps are timed with self.dance_step_start so the main loop keeps
    #     ticking and 'q' remains responsive.
    #   - DEG_PER_SECOND_ROTATE comes from dr_calibration.py.
    # -----------------------------------------------------------------------

    @property
    def _dance_steps(self):
        """
        Build the dance step list once. Each entry is (drive_command, duration).
        Four sets of [+12 right, -24 left, +12 right] -- tighter swings around
        center read more like dancing than the original wide arcs. Shorter
        pauses between steps for snappier rhythm.
        """
        s = SEARCH_ROTATE_SPEED
        t12 = 12.0 / DEG_PER_SECOND_ROTATE      # time to rotate 12 deg
        t24 = 24.0 / DEG_PER_SECOND_ROTATE      # time to rotate 24 deg
        pause = 0.05
        right = ( s,  s, -s, -s)                # rotate clockwise
        left  = (-s, -s,  s,  s)                # rotate counterclockwise
        stop  = ( 0,  0,  0,  0)
        one_set = [
            (right, t12),   #  0 -> +12
            (stop,  pause),
            (left,  t24),   # +12 -> -12
            (stop,  pause),
            (right, t12),   # -12 -> 0
            (stop,  pause),
        ]
        return one_set * 4

    def _run_arrived(self, target):
        if self.dance_complete:
            return  # terminal -- nothing more to do

        steps = self._dance_steps

        # First entry: kick off the dance
        if self.dance_step == 0 and self.dance_step_start == 0.0:
            self.dance_step_start = time.time()
            self.dance_led_last   = time.time()
            self.dance_led_on     = True
            self.robot.Ctrl_WQ2812_ALL(1, 1)   # green on

        # LED flash (~2.5 Hz -- snappy but well below the ~3 Hz photosensitive
        # epilepsy threshold).
        if time.time() - self.dance_led_last >= 0.20:
            self.dance_led_on  = not self.dance_led_on
            self.dance_led_last = time.time()
            if self.dance_led_on:
                self.robot.Ctrl_WQ2812_ALL(1, 1)   # green on
            else:
                self.robot.Ctrl_WQ2812_ALL(0, 0)   # off

        # Step sequencer
        cmd, duration = steps[self.dance_step]
        if time.time() - self.dance_step_start >= duration:
            self.dance_step += 1
            self.dance_step_start = time.time()
            if self.dance_step >= len(steps):
                # Dance done -- stop motors, kill LEDs, latch terminal
                self._stop()
                self._leds_off()
                self.dance_complete = True
                print('[ARRIVED] Dance complete. Press Q to exit.')
                return
            cmd, duration = steps[self.dance_step]

        self._drive(*cmd)

    # -----------------------------------------------------------------------
    # MAIN UPDATE -- called once per frame from main()
    # -----------------------------------------------------------------------

    def update(self, frame):
        self._prune_blacklist()
        target = find_color_block(frame, self.color_key, self.blacklist)

        if   self.state == SEARCHING:         self._run_searching(target)
        elif self.state == TRACKING:          self._run_tracking(target)
        elif self.state == CONFIRMING:        self._run_confirming(target)
        elif self.state == APPROACHING:       self._run_approaching(target)
        elif self.state == SCANNING_OBSTACLE: self._run_scanning_obstacle(target)
        elif self.state == AVOIDING:          self._run_avoiding(target)
        elif self.state == ARRIVED:           self._run_arrived(target)

        return target


# ===========================================================================
# DISPLAY
# ===========================================================================

def draw_overlay(frame, target, tracker):
    """
    Draws all HUD elements:
      - State label (color-coded by state)
      - Target bounding box + center dot + offset line
      - Blacklisted regions (faint red circle + countdown)
      - CONFIRMING prompt (semi-transparent center panel)
      - Dead reckoning readout during obstacle handling
    """
    font       = cv2.FONT_HERSHEY_SIMPLEX
    state_col  = STATE_COLORS.get(tracker.state, (255, 255, 255))

    # Frame center crosshair
    cv2.drawMarker(frame, (FRAME_CX, FRAME_CY),
                   (255, 255, 255), cv2.MARKER_CROSS, 20, 2)

    # Blacklisted zones -- faint red circle + remaining seconds
    now = time.time()
    for bx, by, exp in tracker.blacklist:
        if exp > now:
            remaining = int(exp - now)
            cv2.circle(frame, (bx, by), BLACKLIST_RADIUS_PX, (0, 0, 160), 1)
            cv2.putText(frame, f'{remaining}s',
                        (bx - 12, by + 4), font, 0.4, (0, 0, 160), 1)

    # Target annotation
    if target:
        cv2.rectangle(frame,
                      (target['x'], target['y']),
                      (target['x'] + target['w'], target['y'] + target['h']),
                      tracker.box_color, 2)
        cv2.putText(frame, tracker.color_name.upper(),
                    (target['x'], target['y'] - 10),
                    font, 0.65, tracker.box_color, 2)
        cv2.circle(frame, (target['cx'], target['cy']), 6, (0, 255, 255), -1)
        cv2.line(frame, (FRAME_CX, FRAME_CY),
                 (target['cx'], target['cy']), (255, 255, 0), 2)
        cv2.putText(frame,
                    f'Offset  X:{target["offset_x"]:+d}  Y:{target["offset_y"]:+d}',
                    (10, FRAME_HEIGHT - 40), font, 0.6, (0, 255, 255), 2)
        cv2.putText(frame,
                    f'Area: {target["area_ratio"] * 100:.1f}% of frame',
                    (10, FRAME_HEIGHT - 15), font, 0.6, (0, 255, 255), 2)
    else:
        cv2.putText(frame, 'No target',
                    (10, FRAME_HEIGHT - 15), font, 0.6, (80, 80, 80), 2)

    # CONFIRMING overlay -- semi-transparent panel in the lower-center
    if tracker.state == CONFIRMING:
        panel = frame.copy()
        cv2.rectangle(panel, (60, 340), (580, 430), (15, 15, 15), -1)
        cv2.addWeighted(panel, 0.78, frame, 0.22, 0, frame)
        cv2.putText(frame, 'Is this the correct target?',
                    (105, 375), font, 0.82, (0, 255, 150), 2)
        cv2.putText(frame, '[Y] Confirm      [N] Reject & rescan',
                    (95,  415), font, 0.62, (200, 200, 200), 2)

    # ARRIVED banner -- big, unmistakable
    if tracker.state == ARRIVED:
        panel = frame.copy()
        cv2.rectangle(panel, (60, 180), (580, 300), (10, 60, 10), -1)
        cv2.addWeighted(panel, 0.78, frame, 0.22, 0, frame)
        cv2.putText(frame, 'ARRIVED AT TARGET',
                    (110, 235), font, 1.1, (0, 255, 0), 3)
        msg = 'Performing victory dance...' if not tracker.dance_complete \
              else 'Press Q to exit.'
        cv2.putText(frame, msg, (130, 275), font, 0.65, (220, 255, 220), 2)

    # State label + tracked color
    cv2.putText(frame, f'STATE: {tracker.state}',
                (10, 30), font, 0.8, state_col, 2)
    cv2.putText(frame, f'Tracking: {tracker.color_name}',
                (10, 58), font, 0.6, tracker.box_color, 2)

    # Dead reckoning readout
    if tracker.state in (AVOIDING, SCANNING_OBSTACLE):
        dr = tracker.dr
        cv2.putText(frame,
                    f'DR  pos:({dr.x:.0f},{dr.y:.0f})cm  hdg:{dr.heading:.0f}deg',
                    (10, 85), font, 0.55, (200, 100, 0), 2)

    # ----------------------------------------------------------------------
    # OBSTACLE-HANDLING DIAGNOSTIC HUD
    # During scanning / recentering / avoiding, show a big, unambiguous
    # readout of the most recent sonar scan and the chosen strafe direction.
    # This lets you visually confirm whether the bot's physical motion
    # matches the displayed direction, without watching the terminal.
    # ----------------------------------------------------------------------
    if tracker.state in (SCANNING_OBSTACLE, AVOIDING):
        left  = tracker.scan_left_dist
        right = tracker.scan_right_dist

        # Background panel for legibility
        panel = frame.copy()
        cv2.rectangle(panel, (FRAME_WIDTH - 280, 100),
                              (FRAME_WIDTH - 10,  220), (15, 15, 15), -1)
        cv2.addWeighted(panel, 0.78, frame, 0.22, 0, frame)

        # Left / right clearance readings (color = green if far, red if near)
        def clear_color(d):
            if d >= OBSTACLE_DISTANCE_CM * 1.5: return (0, 255, 0)
            if d >= OBSTACLE_DISTANCE_CM:        return (0, 255, 255)
            return (0, 80, 255)

        cv2.putText(frame, f'L: {left:5.0f} cm',
                    (FRAME_WIDTH - 270, 130), font, 0.7, clear_color(left), 2)
        cv2.putText(frame, f'R: {right:5.0f} cm',
                    (FRAME_WIDTH - 270, 158), font, 0.7, clear_color(right), 2)

        # Big arrow showing chosen avoid direction
        if tracker.avoid_direction > 0:
            dir_text  = '-> RIGHT'
            dir_color = (0, 255, 0) if right > left else (0, 80, 255)
        else:
            dir_text  = 'LEFT <-'
            dir_color = (0, 255, 0) if left > right else (0, 80, 255)
        cv2.putText(frame, dir_text,
                    (FRAME_WIDTH - 270, 198), font, 0.95, dir_color, 3)

    cv2.putText(frame, 'Q: quit',
                (FRAME_WIDTH - 80, 30), font, 0.5, (200, 200, 200), 1)
    return frame


# ===========================================================================
# ENTRY POINT
# ===========================================================================

def main():
    robot = Raspbot()

    cap = cv2.VideoCapture(CAMERA_INDEX)

    # Order matters: set the pixel format (MJPG) BEFORE resolution and FPS,
    # otherwise OpenCV may pick uncompressed YUYV which is bandwidth-limited
    # to ~10-15 fps at 640x480 over USB. MJPG is JPEG-compressed at the
    # sensor and easily sustains 30-60 fps on the same hardware.
    cap.set(cv2.CAP_PROP_FOURCC,       cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          TARGET_FPS)

    # Cap the internal frame buffer at 1. By default OpenCV buffers 3-5
    # frames; if our loop runs slower than the camera, those frames pile up
    # and every cap.read() returns a stale frame from hundreds of ms ago.
    # That manifests as the bot reacting to where the target *was*, not
    # where it is. Buffer=1 makes each read return the latest available frame.
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

    # Wrap the capture in a threaded reader so frame grabs never block the
    # main loop, and so the camera buffer drains continuously even when
    # state runners call time.sleep() (e.g., during obstacle scanning).
    camera = ThreadedCamera(cap)

    if not cap.isOpened():
        print('ERROR: Could not open camera.')
        del robot
        return

    # --- Color selection (blocks until user picks 1-8 or presses Q) ---
    color_key = run_color_selection(camera)
    if color_key is None:
        camera.stop()
        cv2.destroyAllWindows()
        del robot
        return

    # --- Build tracker with the chosen color ---
    tracker = ColorBlockTracker(robot, color_key)
    print(f'Tracking: {COLORS[color_key][0]}')
    print('Y = confirm target  |  N = reject & rescan  |  Q = quit')

    # Holds the last target seen while in CONFIRMING, so reject_target()
    # has coordinates to blacklist even if the target briefly flickers.
    confirming_target = None

    fps_counter = 0
    fps_display = 0
    fps_timer   = time.time()

    # --- Display throttle ---
    # cv2.imshow rendered through VNC is expensive (~30-50ms per frame on a
    # Pi 5). Detection runs every frame for full responsiveness, but the
    # window only updates every Nth frame. Set DISPLAY_EVERY_N=1 if you have
    # a real HDMI display attached -- the cost vanishes there.
    DISPLAY_EVERY_N = 3
    display_counter = 0

    while True:
        ret, frame = camera.read()
        if not ret:
            time.sleep(0.005)
            continue

        target = tracker.update(frame)

        # Keep confirming_target fresh while we're waiting for Y/N.
        # Once the state leaves CONFIRMING (by any path), clear it.
        if tracker.state == CONFIRMING:
            if target:
                confirming_target = target
        else:
            confirming_target = None

        fps_counter += 1
        if time.time() - fps_timer >= 1.0:
            fps_display = fps_counter
            fps_counter = 0
            fps_timer   = time.time()

        # --- Throttled display ---
        # Always render on CONFIRMING frames so the user can react to the
        # prompt promptly, even if the throttle would normally skip them.
        display_counter += 1
        should_display = (display_counter % DISPLAY_EVERY_N == 0
                          or tracker.state == CONFIRMING)

        if should_display:
            frame = draw_overlay(frame, target, tracker)
            cv2.putText(frame, f'FPS: {fps_display}',
                        (FRAME_WIDTH - 100, FRAME_HEIGHT - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow('Color Block Tracker', frame)

        # waitKey(1) is required even on non-display frames because it's
        # what services the OpenCV window event queue and reads keystrokes.
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('y') and tracker.state == CONFIRMING:
            tracker.confirm_target()
        elif key == ord('n') and tracker.state == CONFIRMING:
            if confirming_target:
                tracker.reject_target(confirming_target)

    for i in range(4): robot.Ctrl_Muto(i, 0)
    robot.Ctrl_WQ2812_ALL(0, 0)    # LED bar off
    robot.Ctrl_Ulatist_Switch(0)   # Power off ultrasonic sensor
    camera.stop()
    cv2.destroyAllWindows()
    del robot
    print('Tracker closed.')


if __name__ == '__main__':
    main()