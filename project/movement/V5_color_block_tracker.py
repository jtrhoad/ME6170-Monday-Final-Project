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
  AVOIDING    -- Strafe sideways then drive forward past obstacle.
                 Returns to SEARCHING for visual reacquisition (no lane return).
  REACQUIRING -- (legacy) Rotate toward DR-estimated bearing. No longer used
                 by the normal flow; kept for reference.
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
  Calibrate DEG_PER_SECOND_ROTATE, CM_PER_SECOND_FORWARD, CM_PER_SECOND_STRAFE
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
# CONFIGURATION
# ===========================================================================

# --- Camera ---
CAMERA_INDEX = 0
FRAME_WIDTH  = 640
FRAME_HEIGHT = 480
TARGET_FPS   = 60                # USB cameras have a hard ceiling. Setting
                                 # this to 600 won't make the camera go faster
                                 # than its physical mode supports -- it caps
                                 # to whatever the sensor offers <=60.

# --- Servo ---
SERVO_PAN         = 1
SERVO_TILT        = 2
PAN_CENTER        = 72
TILT_CENTER       = 25
PAN_MIN,  PAN_MAX  =  5, 175    # Hard angle limits -- never exceeded
TILT_MIN, TILT_MAX =  0,  95

# Proportional gain: degrees of servo movement per pixel of error.
# Lower = smoother but slower. Higher = faster but more jitter.
PAN_GAIN     = 0.03
TILT_GAIN    = 0.03

# Max servo movement per frame -- prevents lurching on large sudden errors.
MAX_STEP_DEG = 2.0

# Ignore offsets smaller than this -- stops jitter when nearly centered.
DEADBAND_PX  = 5

# How close (in pixels) the target must be to frame center on BOTH axes
# before TRACKING declares the bot centered and advances to CONFIRMING.
# Wider than DEADBAND_PX because the chassis is much coarser than the
# camera servos -- gear backlash and motor deadzone make sub-10px body
# centering effectively impossible. 25 px is roughly +/- 4% of frame width.
TRACK_CENTERED_PX = 25

# Minimum motor command magnitude that actually produces movement.
# Below this, the wheels stall instead of rotating. Used as a floor for
# the tracking-yaw PID output so that tiny corrections still move the bot.
MIN_TRACK_ROTATE_SPEED = 25

# --- Detection ---
MIN_CONTOUR_AREA = 1500          # px^2 -- filters noise and far-away blobs

# --- Blacklist (sticky -- follows the rejected object as the bot rotates) ---
# When a target is rejected, the rejected contour's center is recorded as a
# blacklist entry. Each frame, find_color_block tries to snap each entry to
# the nearest qualifying contour center within BLACKLIST_TRACK_RADIUS_PX --
# this lets the entry "ride" the rejected blob as it slides across the frame
# during rotation. Contours within BLACKLIST_RADIUS_PX of a (now-updated)
# entry are excluded from selection.
BLACKLIST_DURATION       = 5.0       # seconds the region stays blocked
BLACKLIST_RADIUS_PX      = 100       # pixel radius of the exclusion zone
BLACKLIST_TRACK_RADIUS_PX = 140      # max per-frame jump for entry-tracking

# --- Approach ---
APPROACH_AREA_STOP = 0.25        # legacy fallback -- arrival now uses sonar
APPROACH_SPEED     = 35          # was 50 -- slower so vision tracking can
                                 # keep up at lower frame rates
ARRIVAL_DISTANCE_CM = 20.0       # stop when sonar reads <= this distance
                                 # Smaller than OBSTACLE_DISTANCE_CM, so there
                                 # is a 5 cm window (20-25 cm) where sonar
                                 # would otherwise treat the target itself as
                                 # an obstacle. _run_approaching suppresses the
                                 # obstacle check when a target is currently
                                 # visible and roughly centered -- see
                                 # TARGET_LOCK_PX below.
TARGET_LOCK_PX      = 80         # if a target is within this many pixels of
                                 # frame-center horizontally, we trust that
                                 # whatever sonar sees IS the target, not an
                                 # obstacle, and skip the obstacle check.
ARRIVAL_MIN_AREA    = 0.08       # target must fill at least this fraction of
                                 # the frame to count as "arrived." Prevents
                                 # false arrivals when an unrelated object
                                 # crosses sonar at <15 cm while the real
                                 # target is still far away. At 15 cm a typical
                                 # 5-10 cm target should easily exceed this.
                                 # Tune up if false arrivals happen, down if
                                 # real arrivals are missed.

# --- Yaw PID (keeps target horizontally centered while driving forward) ---
# Error  = target.offset_x in pixels (positive = target is right of center)
# Output = wheel-speed bias added to LEFT wheels and subtracted from RIGHT
#          (positive bias rotates the bot clockwise toward a right-side target)
# Tune Kp first; only add Ki/Kd if you see steady-state drift or oscillation.
YAW_KP             = 0.15
YAW_KI             = 0.01
YAW_KD             = 0.05
YAW_OUTPUT_LIMIT   = 35           # max wheel-speed bias (units of motor speed)
YAW_INTEGRAL_LIMIT = 200          # anti-windup cap on the integral term
YAW_DEADBAND_PX    = 8            # ignore tiny errors so the bot doesn't twitch

# --- Obstacle Avoidance ---
OBSTACLE_DISTANCE_CM   = 20.0     # bumped from 20 to maintain buffer over
                                  # the new 20cm arrival distance
AVOID_SIDE_SPEED       = 60
AVOID_SIDE_DURATION    = 1.45     # was 0.95 -- strafe further so the bot
                                  # has more clearance when re-aligning to
                                  # the target after the forward phase.
                                  # At CM_PER_SECOND_STRAFE = 28.9 this is
                                  # roughly 42 cm of lateral movement.
AVOID_FORWARD_DURATION = 1.0
AVOID_SPEED            = 50

# --- Obstacle Scanning (rotate body, read sonar both sides) ---
SCAN_ROTATE_ANGLE_DEG  = 90.0     # how far to swing each side
SCAN_SETTLE_TIME       = 0.25     # let sensor stabilize after rotation stops
SCAN_SAMPLES           = 3        # average this many sonar reads per side

# If both scan sides come back within this distance, the bot is "trapped"
# (obstacles within obstacle range on both flanks). It rotates 180 and
# drives forward (which now points away from the obstacles) to escape.
TRAPPED_THRESHOLD_CM = OBSTACLE_DISTANCE_CM   # both <= obstacle range = trapped

# --- Escape (180 + drive) when trapped ---
ESCAPE_FORWARD_DURATION = 1.5    # seconds of forward drive after the 180
ESCAPE_FORWARD_SPEED    = 40

# --- Post-scan vision recentering ---
# After SCANNING_OBSTACLE returns the body to "forward", motor inertia and
# calibration drift can leave the chassis 20-50 deg off the original heading.
# Before committing to a strafe direction, we re-center on the target using
# the body PID so the strafe is actually perpendicular to the line of sight.
# If the target isn't visible (e.g., obstacle is blocking the camera too),
# fall through to AVOIDING after this many seconds.
RECENTER_TIMEOUT_S = 2.0

# --- Search ---
# Rotation speed and step size for in-place rotations (search, scan, dance,
# recenter). We deliberately match dr_calibration.py's hardcoded
# SEARCH_ROTATE_SPEED = 40 so DEG_PER_SECOND_ROTATE below is the measured
# value at this exact PWM, with no linear-scaling approximation needed.
#
# At 126 deg/sec, a 0.10 s step covers ~13 deg -- well inside the camera's
# ~60 deg FOV so a target appears in 4-5 consecutive frames.
SEARCH_ROTATE_SPEED = 40
SEARCH_ROTATE_STEP  = 0.10        # seconds per rotate step before rechecking
SEARCH_MAX_STEPS    = 35          # ~455 deg total, > 1 full rotation

# --- Dead Reckoning (calibrate on actual surface) ---
# --- Dead Reckoning (calibrate on actual surface) ---
# DEG_PER_SECOND_ROTATE: measured at SEARCH_ROTATE_SPEED = 40.
# Latest measurement: compass heading swept from 105 to 357 (252 deg)
# in 2.0 s -> 126 deg/sec. Rerun dr_calibration.py Test 1 if you change
# SEARCH_ROTATE_SPEED above, since this constant is rate-at-that-PWM.
DEG_PER_SECOND_ROTATE = 126.0
CM_PER_SECOND_FORWARD = 35.25
CM_PER_SECOND_STRAFE  = 28.90

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

SEARCHING            = 'SEARCHING'
TRACKING             = 'TRACKING'
CONFIRMING           = 'CONFIRMING'
APPROACHING          = 'APPROACHING'
SCANNING_OBSTACLE    = 'SCANNING_OBSTACLE'
RECENTERING_FOR_AVOID = 'RECENTERING_FOR_AVOID'
AVOIDING             = 'AVOIDING'
ESCAPE_BACKWARD      = 'ESCAPE_BACKWARD'
REACQUIRING          = 'REACQUIRING'
ARRIVED              = 'ARRIVED'

FRAME_CX   = FRAME_WIDTH  // 2   # 320
FRAME_CY   = FRAME_HEIGHT // 2   # 240
FRAME_AREA = FRAME_WIDTH * FRAME_HEIGHT

# On-screen state label colors (BGR)
STATE_COLORS = {
    SEARCHING:             (100, 100, 100),
    TRACKING:              (0,   200, 255),
    CONFIRMING:            (0,   255, 150),
    APPROACHING:           (0,   200, 0),
    SCANNING_OBSTACLE:     (0,   140, 255),
    RECENTERING_FOR_AVOID: (0,   200, 255),
    AVOIDING:              (0,   80,  255),
    ESCAPE_BACKWARD:       (0,   0,   255),
    REACQUIRING:           (200, 100, 0),
    ARRIVED:               (0,   255, 0),
}

# LED bar color codes per Yahboom Ctrl_WQ2812_ALL:
#   0=red  1=green  2=blue  3=yellow  4=purple  5=cyan  6=white
# A value of None means "leave LEDs in their current managed state"
# (used by ARRIVED so the dance can flash without being overridden each frame).
STATE_LED_COLORS = {
    SEARCHING:             5,   # cyan   -- scanning
    TRACKING:              3,   # yellow -- locked, centering
    CONFIRMING:            6,   # white  -- waiting on user
    APPROACHING:           1,   # green  -- driving toward target
    SCANNING_OBSTACLE:     3,   # yellow -- caution, looking around
    RECENTERING_FOR_AVOID: 3,   # yellow -- still in obstacle handling
    AVOIDING:              0,   # red    -- maneuvering around obstacle
    ESCAPE_BACKWARD:       0,   # red    -- trapped escape maneuver
    REACQUIRING:           4,   # purple -- hunting for lost target
    ARRIVED:               None,
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

def compute_servo_step(error_px, gain, max_step):
    """
    Convert a pixel error into a servo degree step.

    1. Scale by gain (error -> degrees).
    2. Cap at max_step so large errors don't cause sudden lurches.
    3. Preserve sign so direction is correct.

    Example: error=80px, gain=0.03, max_step=2.0
      raw = 80 * 0.03 = 2.4 -> clamped to 2.0 degrees
    """
    raw = error_px * gain
    return float(np.clip(raw, -max_step, max_step))


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

    def strafe(self, direction, duration):
        """direction: +1 = right, -1 = left"""
        dist = CM_PER_SECOND_STRAFE * duration * direction
        rad  = math.radians(self.heading + 90)
        self.x += dist * math.sin(rad)
        self.y += dist * math.cos(rad)
        dbg(f'[DR] Strafe {dist:+.1f}cm  pos ({self.x:.1f}, {self.y:.1f})')

    @property
    def estimated_block_bearing(self):
        """
        Angle (degrees) to rotate in order to face the block's last known
        position from our current estimated position and heading.
        """
        bearing = math.degrees(math.atan2(-self.x, -self.y))
        return bearing - self.heading


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
        self.avoid_phase      = 0
        self.avoid_direction  = +1   # +1 strafe right, -1 strafe left
        self.last_action_time = time.time()

        # Yaw correction PID for APPROACHING (keeps target horizontally centered)
        self.yaw_pid = PIDController(
            kp=YAW_KP, ki=YAW_KI, kd=YAW_KD,
            output_limit=YAW_OUTPUT_LIMIT,
            integral_limit=YAW_INTEGRAL_LIMIT,
            deadband=YAW_DEADBAND_PX,
        )

        # Obstacle scanning state (now phased / non-blocking)
        self.scan_phase      = 0
        self.scan_phase_start = 0.0
        self.scan_left_dist  = 999.0
        self.scan_right_dist = 999.0

        # Escape (180 + drive forward) state when trapped
        self.escape_phase       = 0
        self.escape_phase_start = 0.0

        # Arrival dance state
        self.dance_step          = 0
        self.dance_step_start    = 0.0
        self.dance_led_last      = 0.0
        self.dance_led_on        = False
        self.dance_complete      = False

        # Each entry: (cx, cy, expiry_time)
        # Contours near a live entry are skipped in find_color_block()
        self.blacklist = []

        # Cache of last motor command per wheel -- _drive skips I2C writes
        # for unchanged values to reduce bus traffic and shave loop time.
        # Sentinel "None" means "force the next write."
        self._last_motor_cmd = [None, None, None, None]

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

        OPTIMIZATION: skip the I2C write if the new command equals the last
        one we sent to that motor. Saves ~1-2 ms per frame in steady-state
        driving (every frame in APPROACHING the four motor commands are
        identical, so we'd be re-sending the same value unnecessarily).
        Also reduces I2C bus contention which mildly helps the wheel-lockup
        symptom by leaving more bandwidth for retries on the actual changes.
        """
        new = (fl, fr, rl, rr)
        for i, val in enumerate(new):
            if self._last_motor_cmd[i] != val:
                self.robot.Ctrl_Muto(i, val)
                self._last_motor_cmd[i] = val

    def _stop(self):
        """Stop all four wheels. Bypasses the cache so a stop always reaches
        the hardware even if the prior command was already 0 -- safety wins
        over the small I2C cost."""
        for i in range(4):
            self.robot.Ctrl_Muto(i, 0)
            self._last_motor_cmd[i] = 0

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
        loop endlessly through SCANNING_OBSTACLE -> ESCAPE_BACKWARD.
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

    def _rotate_blocking(self, direction, degrees):
        """
        Rotate the body in place by an exact angle, blocking until complete.
        DEPRECATED in favor of phased non-blocking rotations -- left in place
        for any callers that still need it. Blocks the main loop, so the
        camera buffer fills with stale frames during the rotation.
        direction: +1 = clockwise (right), -1 = counterclockwise (left)
        """
        duration = degrees / DEG_PER_SECOND_ROTATE
        speed    = SEARCH_ROTATE_SPEED
        self._drive( speed * direction,  speed * direction,
                    -speed * direction, -speed * direction)
        time.sleep(duration)
        self._stop()
        self.dr.rotate(direction, duration)

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
    def _rotation_duration_for(degrees):
        """How many seconds at SEARCH_ROTATE_SPEED to rotate the given angle."""
        return degrees / DEG_PER_SECOND_ROTATE

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
            if new_state in (TRACKING, APPROACHING, RECENTERING_FOR_AVOID):
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
            step      = compute_servo_step(target['offset_y'], TILT_GAIN, MAX_STEP_DEG)
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
    # Rotate bot clockwise in SEARCH_ROTATE_STEP increments.
    # Stops and transitions to TRACKING the moment a blob is detected.
    # After a full 360, nudges forward and restarts scan.
    # -----------------------------------------------------------------------

    def _run_searching(self, target):
        if target:
            self._stop()
            self.search_steps = 0
            self.dr.reset()
            self._set_state(TRACKING)
            return

        if self._elapsed() >= SEARCH_ROTATE_STEP:
            # In-place clockwise: left side (motors 0,1) forward, right side (motors 2,3) backward
            # Confirmed vs dr_calibration.py: Muto(0,+S), Muto(1,+S), Muto(2,-S), Muto(3,-S)
            self._drive( SEARCH_ROTATE_SPEED,  SEARCH_ROTATE_SPEED,
                        -SEARCH_ROTATE_SPEED, -SEARCH_ROTATE_SPEED)
            self.dr.rotate(+1, SEARCH_ROTATE_STEP)
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
    # SCANNING_OBSTACLE  (non-blocking, phased)
    # Sonar pair is fixed forward, so we rotate the *body* to point them at
    # the left side and then the right side. After both readings the bot
    # rotates back to its original heading and either:
    #   - picks the side with greater clearance and goes to RECENTERING_FOR_AVOID, or
    #   - if BOTH sides are within TRAPPED_THRESHOLD_CM, declares trapped and
    #     transitions to ESCAPE_BACKWARD (180 + drive away).
    #
    # Phases (each frame just checks elapsed time, never sleeps):
    #   0  start LEFT rotation
    #   1  rotating LEFT  -- wait
    #   2  settle, sample LEFT sonar, start RIGHT rotation
    #   3  rotating RIGHT -- wait
    #   4  settle, sample RIGHT sonar, start return-LEFT rotation
    #   5  rotating BACK to center -- wait
    #   6  decide: trapped -> ESCAPE_BACKWARD, otherwise pick side -> RECENTER
    #
    # The cost of all this phase bookkeeping is that the main loop keeps
    # ticking, the camera buffer drains, and the FPS counter stays high.
    # -----------------------------------------------------------------------

    def _run_scanning_obstacle(self, target):
        elapsed = time.time() - self.scan_phase_start

        if self.scan_phase == 0:
            # Start LEFT rotation
            print(f'[SCAN/0] Starting LEFT rotation '
                  f'({SCAN_ROTATE_ANGLE_DEG:.0f} deg, '
                  f'~{self._rotation_duration_for(SCAN_ROTATE_ANGLE_DEG):.2f}s)')
            self._start_rotation(-1)
            self.scan_phase       = 1
            self.scan_phase_start = time.time()

        elif self.scan_phase == 1:
            # Wait for LEFT rotation to complete
            if elapsed >= self._rotation_duration_for(SCAN_ROTATE_ANGLE_DEG):
                self._stop()
                self.dr.rotate(-1, self._rotation_duration_for(SCAN_ROTATE_ANGLE_DEG))
                print(f'[SCAN/1] LEFT rotation done (took {elapsed:.2f}s); '
                      f'settling for {SCAN_SETTLE_TIME}s')
                self.scan_phase       = 2
                self.scan_phase_start = time.time()

        elif self.scan_phase == 2:
            # Settle, sample LEFT, start RIGHT rotation
            if elapsed >= SCAN_SETTLE_TIME:
                # Print individual samples so we can see whether the sensor
                # is returning consistent values or wild noise
                samples = []
                for _ in range(SCAN_SAMPLES):
                    samples.append(self._read_sonar())
                    time.sleep(0.02)
                valid = [s for s in samples if s < 999.0]
                self.scan_left_dist = (sum(valid) / len(valid)
                                       if valid else 999.0)
                print(f'[SCAN/2] LEFT samples: {[round(s,1) for s in samples]} '
                      f'-> avg {self.scan_left_dist:.1f}cm')
                print(f'[SCAN/2] Starting RIGHT rotation '
                      f'({2*SCAN_ROTATE_ANGLE_DEG:.0f} deg)')
                self._start_rotation(+1)
                self.scan_phase       = 3
                self.scan_phase_start = time.time()

        elif self.scan_phase == 3:
            # Wait for RIGHT rotation to complete (180 deg total swing)
            if elapsed >= self._rotation_duration_for(2 * SCAN_ROTATE_ANGLE_DEG):
                self._stop()
                self.dr.rotate(+1, self._rotation_duration_for(2 * SCAN_ROTATE_ANGLE_DEG))
                print(f'[SCAN/3] RIGHT rotation done (took {elapsed:.2f}s); '
                      f'settling for {SCAN_SETTLE_TIME}s')
                self.scan_phase       = 4
                self.scan_phase_start = time.time()

        elif self.scan_phase == 4:
            # Settle, sample RIGHT, start return rotation
            if elapsed >= SCAN_SETTLE_TIME:
                samples = []
                for _ in range(SCAN_SAMPLES):
                    samples.append(self._read_sonar())
                    time.sleep(0.02)
                valid = [s for s in samples if s < 999.0]
                self.scan_right_dist = (sum(valid) / len(valid)
                                        if valid else 999.0)
                print(f'[SCAN/4] RIGHT samples: {[round(s,1) for s in samples]} '
                      f'-> avg {self.scan_right_dist:.1f}cm')
                print(f'[SCAN/4] Starting return rotation '
                      f'({SCAN_ROTATE_ANGLE_DEG:.0f} deg LEFT)')
                self._start_rotation(-1)
                self.scan_phase       = 5
                self.scan_phase_start = time.time()

        elif self.scan_phase == 5:
            # Wait for return rotation
            if elapsed >= self._rotation_duration_for(SCAN_ROTATE_ANGLE_DEG):
                self._stop()
                self.dr.rotate(-1, self._rotation_duration_for(SCAN_ROTATE_ANGLE_DEG))
                print(f'[SCAN/5] Return rotation done (took {elapsed:.2f}s)')
                self.scan_phase = 6

        elif self.scan_phase == 6:
            # Decide and transition
            self._scan_decide_next()

    def _scan_decide_next(self):
        """
        After all sonar samples are in and the body is back to center, decide:
          - both sides blocked -> ESCAPE_BACKWARD
          - otherwise pick the clearer side, go to RECENTERING_FOR_AVOID
        """
        left  = self.scan_left_dist
        right = self.scan_right_dist

        # --- Trapped check ---
        if left <= TRAPPED_THRESHOLD_CM and right <= TRAPPED_THRESHOLD_CM:
            print(f'[TRAPPED] L={left:.0f}cm R={right:.0f}cm '
                  f'(both <={TRAPPED_THRESHOLD_CM:.0f}cm) -- escaping backward')
            self.escape_phase       = 0
            self.escape_phase_start = time.time()
            self._set_state(ESCAPE_BACKWARD)
            return

        # --- Pick the side with more clearance ---
        TIE_THRESHOLD_CM = 5.0
        TIE_DEFAULT      = +1   # +1 = right, -1 = left
        diff = abs(left - right)

        if diff < TIE_THRESHOLD_CM:
            self.avoid_direction = TIE_DEFAULT
            side = 'RIGHT' if TIE_DEFAULT > 0 else 'LEFT'
            print(f'[SCAN] L={left:.0f}cm R={right:.0f}cm (tie, defaulting {side})')
        elif left > right:
            self.avoid_direction = -1
            print(f'[SCAN] L={left:.0f}cm R={right:.0f}cm  -> LEFT')
        else:
            self.avoid_direction = +1
            print(f'[SCAN] L={left:.0f}cm R={right:.0f}cm  -> RIGHT')

        self.last_action_time = time.time()
        self._set_state(RECENTERING_FOR_AVOID)

    # -----------------------------------------------------------------------
    # ESCAPE_BACKWARD  (non-blocking, phased)
    # Trapped on both sides. Rotate 180 deg in place, then drive forward
    # (which is now away from the obstacles), then return to SEARCHING so
    # the bot can spin to find the target visually from the new position.
    #
    # We can't do a literal "back up" without a rear sensor, and rotating
    # first lets the forward-facing sonar verify the new path is clear if
    # it isn't (the next OBSTACLE check will catch it).
    #
    # Phases:
    #   0  start 180 rotation
    #   1  rotating -- wait
    #   2  start forward drive
    #   3  driving -- wait
    #   4  done -> SEARCHING
    # -----------------------------------------------------------------------

    def _run_escape_backward(self, target):
        elapsed = time.time() - self.escape_phase_start

        if self.escape_phase == 0:
            # Start 180 deg clockwise rotation (direction is arbitrary;
            # clockwise picked for consistency with SEARCHING which also
            # rotates clockwise)
            self._start_rotation(+1)
            self.escape_phase       = 1
            self.escape_phase_start = time.time()

        elif self.escape_phase == 1:
            if elapsed >= self._rotation_duration_for(180.0):
                self._stop()
                self.dr.rotate(+1, self._rotation_duration_for(180.0))
                self.escape_phase       = 2
                self.escape_phase_start = time.time()

        elif self.escape_phase == 2:
            # Start forward drive (away from original obstacles)
            self._drive(ESCAPE_FORWARD_SPEED, ESCAPE_FORWARD_SPEED,
                        ESCAPE_FORWARD_SPEED, ESCAPE_FORWARD_SPEED)
            self.escape_phase       = 3
            self.escape_phase_start = time.time()

        elif self.escape_phase == 3:
            # Mid-escape: if a NEW obstacle appears in front, abort early
            # by stopping and going to SEARCHING. Bot will rescan from there.
            dist = self._read_sonar()
            if dist < OBSTACLE_DISTANCE_CM:
                print(f'[ESCAPE] Aborted -- obstacle at {dist:.1f}cm '
                      f'after rotation, handing off to SEARCHING')
                self._stop()
                self.dr.move_forward(elapsed)   # log partial distance
                self.dr.reset()
                self._set_state(SEARCHING)
                return
            if elapsed >= ESCAPE_FORWARD_DURATION:
                self._stop()
                self.dr.move_forward(ESCAPE_FORWARD_DURATION)
                self.dr.reset()
                print('[ESCAPE] Complete -- returning to SEARCHING')
                self._set_state(SEARCHING)

    # -----------------------------------------------------------------------
    # RECENTERING_FOR_AVOID
    # Vision-based heading correction after SCANNING_OBSTACLE. The dead-
    # reckoned rotations during scanning are subject to motor inertia and
    # speed-vs-time calibration drift, so by the time we return to "center"
    # the chassis can be 20-50 deg off the original target heading. If we
    # strafed at that point, we'd move at the wrong angle and either clip
    # the obstacle or miss the target re-acquisition.
    #
    # This state runs the body PID against the target's horizontal pixel
    # offset (same logic as TRACKING) until the target is centered, then
    # starts the strafe. If the target isn't visible (camera blocked, or
    # we're way off heading), it falls through to AVOIDING after
    # RECENTER_TIMEOUT_S so the bot doesn't sit forever.
    # -----------------------------------------------------------------------

    def _start_avoiding(self):
        """Prep state and transition into AVOIDING."""
        self._stop()
        self.avoid_phase      = 0
        self.last_action_time = time.time()
        self._set_state(AVOIDING)

    def _run_recentering_for_avoid(self, target):
        # Hard timeout -- always proceed eventually so the bot can't get
        # stuck spinning here if the target is occluded.
        if self._elapsed() > RECENTER_TIMEOUT_S:
            print(f'[RECENTER] Timeout ({RECENTER_TIMEOUT_S}s) -- '
                  f'proceeding to AVOIDING')
            self._start_avoiding()
            return

        if not target:
            # No target visible. Just hold position and let the timeout
            # carry us forward. Rotating to search here would risk losing
            # whatever heading we have.
            self._stop()
            return

        x_off = target['offset_x']

        # Centered? Done -- start the strafe.
        if abs(x_off) <= TRACK_CENTERED_PX:
            print(f'[RECENTER] Visually centered (offset_x={x_off:+d}px) '
                  f'-- starting AVOIDING')
            self._start_avoiding()
            return

        # Body PID rotation to bring x_off toward zero (same pattern as
        # _run_tracking).
        raw_speed = self.yaw_pid.update(float(x_off))
        speed     = int(raw_speed)
        if 0 < abs(speed) < MIN_TRACK_ROTATE_SPEED:
            speed = MIN_TRACK_ROTATE_SPEED if speed > 0 else -MIN_TRACK_ROTATE_SPEED
        self._drive( speed,  speed, -speed, -speed)

    # -----------------------------------------------------------------------
    # AVOIDING
    # Two-phase Mecanum maneuver, parameterized by self.avoid_direction:
    #   Phase 0: strafe sideways  (clear obstacle laterally)
    #   Phase 1: drive forward    (pass the obstacle)
    # After phase 1 the bot transitions back to SEARCHING, which spins in
    # place to visually reacquire the target. Once found, the normal
    # SEARCHING -> TRACKING -> CONFIRMING -> APPROACHING flow takes over.
    #
    # We deliberately do NOT strafe back to the original "lane" -- the user
    # only cares about reaching the target, not preserving the original
    # straight-line approach path. Returning to the lane wasted time and
    # often re-collided with the obstacle.
    # -----------------------------------------------------------------------

    def _run_avoiding(self, target):
        # If the block reappears mid-maneuver (after we have lateral clearance),
        # skip the rest of the avoid and let TRACKING re-center on it. The
        # phase 0 strafe still completes regardless so we're guaranteed to
        # have moved sideways before this can fire.
        if target and self.avoid_phase > 0:
            self._stop()
            self._set_state(TRACKING)
            return

        elapsed = self._elapsed()
        d       = self.avoid_direction   # +1 = right, -1 = left

        if self.avoid_phase == 0:
            if elapsed < AVOID_SIDE_DURATION:
                # Mecanum strafe RIGHT  : (-S, +S, +S, -S)
                # Mecanum strafe LEFT   : (+S, -S, -S, +S)  -- multiply by -d
                s = AVOID_SIDE_SPEED
                self._drive(-s * d,  s * d,
                             s * d, -s * d)
            else:
                self.dr.strafe(d, AVOID_SIDE_DURATION)
                self._stop()
                self.avoid_phase      = 1
                self.last_action_time = time.time()

        elif self.avoid_phase == 1:
            if elapsed < AVOID_FORWARD_DURATION:
                self._drive(AVOID_SPEED, AVOID_SPEED,
                            AVOID_SPEED, AVOID_SPEED)
            else:
                self.dr.move_forward(AVOID_FORWARD_DURATION)
                self._stop()
                # Done avoiding -- spin in place to visually find the target.
                # Reset DR because the lane geometry no longer matters and we
                # don't want stale heading offsets confusing future scans.
                self.dr.reset()
                self._set_state(SEARCHING)

    # -----------------------------------------------------------------------
    # REACQUIRING
    # Obstacle is behind us. Rotate toward the dead-reckoning estimated bearing
    # until the block reappears or we're close enough to hand off to SEARCHING.
    # -----------------------------------------------------------------------

    def _run_reacquiring(self, target):
        if target:
            self._stop()
            self._set_state(TRACKING)
            return

        bearing = self.dr.estimated_block_bearing
        dbg(f'[REACQUIRE] Estimated bearing: {bearing:+.1f}deg')

        if abs(bearing) > 10:
            direction   = +1 if bearing > 0 else -1
            rotate_time = min(abs(bearing) / DEG_PER_SECOND_ROTATE, 0.5)
            # Rotate: left side forward/back, right side opposite -- confirmed vs dr_calibration
            self._drive(
                 SEARCH_ROTATE_SPEED * direction,  SEARCH_ROTATE_SPEED * direction,
                -SEARCH_ROTATE_SPEED * direction, -SEARCH_ROTATE_SPEED * direction)
            time.sleep(rotate_time)
            self.dr.rotate(direction, rotate_time)
            self._stop()
        else:
            # Close enough to estimated angle -- hand off to SEARCHING
            self.dr.reset()
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

        if   self.state == SEARCHING:             self._run_searching(target)
        elif self.state == TRACKING:              self._run_tracking(target)
        elif self.state == CONFIRMING:            self._run_confirming(target)
        elif self.state == APPROACHING:           self._run_approaching(target)
        elif self.state == SCANNING_OBSTACLE:     self._run_scanning_obstacle(target)
        elif self.state == RECENTERING_FOR_AVOID: self._run_recentering_for_avoid(target)
        elif self.state == AVOIDING:              self._run_avoiding(target)
        elif self.state == ESCAPE_BACKWARD:       self._run_escape_backward(target)
        elif self.state == REACQUIRING:           self._run_reacquiring(target)
        elif self.state == ARRIVED:               self._run_arrived(target)

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
      - Dead reckoning readout during AVOIDING / REACQUIRING
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
    if tracker.state in (AVOIDING, REACQUIRING, SCANNING_OBSTACLE,
                         RECENTERING_FOR_AVOID, ESCAPE_BACKWARD):
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
    if tracker.state in (SCANNING_OBSTACLE, RECENTERING_FOR_AVOID, AVOIDING):
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