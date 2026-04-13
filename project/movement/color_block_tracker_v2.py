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
  TRACKING    -- Blob found; pan/tilt camera to center it. Wheels stopped.
  CONFIRMING  -- Camera centered; ask user Y/N before approaching.
  APPROACHING -- Confirmed; drive forward while keeping camera on target.
  AVOIDING    -- Obstacle detected; navigate around it with dead reckoning.
  REACQUIRING -- Post-obstacle; rotate toward estimated block bearing.

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
import numpy as np
from Raspbot_Lib import Raspbot

# ===========================================================================
# CONFIGURATION
# ===========================================================================

# --- Camera ---
CAMERA_INDEX = 0
FRAME_WIDTH  = 640
FRAME_HEIGHT = 480
TARGET_FPS   = 60

# --- Servo ---
SERVO_PAN         = 1
SERVO_TILT        = 2
PAN_CENTER        = 72
TILT_CENTER       = 25
PAN_MIN,  PAN_MAX  =  5, 175    # Hard angle limits -- never exceeded
TILT_MIN, TILT_MAX =  5,  95

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

# --- Blacklist ---
# When a target is rejected, its frame-center pixel location is blacklisted.
# Any new contour whose center falls within BLACKLIST_RADIUS_PX of a live
# blacklist entry is skipped entirely during detection.
BLACKLIST_DURATION  = 8.0        # seconds the region stays blocked
BLACKLIST_RADIUS_PX = 100        # pixel radius of the exclusion zone

# --- Approach ---
APPROACH_AREA_STOP = 0.25        # legacy fallback -- arrival now uses sonar
APPROACH_SPEED     = 50
ARRIVAL_DISTANCE_CM = 15.0       # stop when sonar reads <= this distance
                                 # Smaller than OBSTACLE_DISTANCE_CM, so there
                                 # is a 5 cm window (15-20 cm) where sonar
                                 # would otherwise treat the target itself as
                                 # an obstacle. _run_approaching suppresses the
                                 # obstacle check when a target is currently
                                 # visible and roughly centered -- see
                                 # TARGET_LOCK_PX below.
TARGET_LOCK_PX      = 80         # if a target is within this many pixels of
                                 # frame-center horizontally, we trust that
                                 # whatever sonar sees IS the target, not an
                                 # obstacle, and skip the obstacle check.
ARRIVAL_MIN_AREA    = 0.1       # target must fill at least this fraction of
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
OBSTACLE_DISTANCE_CM   = 20.0
AVOID_SIDE_SPEED       = 60
AVOID_SIDE_DURATION    = 0.95     # was 1.4 -- reduced to fix ~10-15cm overshoot
AVOID_FORWARD_DURATION = 1.0
AVOID_SPEED            = 50

# --- Obstacle Scanning (rotate body, read sonar both sides) ---
SCAN_ROTATE_ANGLE_DEG  = 90.0     # how far to swing each side
SCAN_SETTLE_TIME       = 0.25     # let sensor stabilize after rotation stops
SCAN_SAMPLES           = 3        # average this many sonar reads per side

# --- Search ---
SEARCH_ROTATE_SPEED = 40
SEARCH_ROTATE_STEP  = 0.3        # seconds per rotate step before rechecking
SEARCH_MAX_STEPS    = 24         # 24 steps * 0.3s ~ 360 degrees

# --- Dead Reckoning (calibrate on actual surface) ---
DEG_PER_SECOND_ROTATE = 101.0
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

SEARCHING         = 'SEARCHING'
TRACKING          = 'TRACKING'
CONFIRMING        = 'CONFIRMING'
APPROACHING       = 'APPROACHING'
SCANNING_OBSTACLE = 'SCANNING_OBSTACLE'
AVOIDING          = 'AVOIDING'
REACQUIRING       = 'REACQUIRING'
ARRIVED           = 'ARRIVED'

FRAME_CX   = FRAME_WIDTH  // 2   # 320
FRAME_CY   = FRAME_HEIGHT // 2   # 240
FRAME_AREA = FRAME_WIDTH * FRAME_HEIGHT

# On-screen state label colors (BGR)
STATE_COLORS = {
    SEARCHING:         (100, 100, 100),
    TRACKING:          (0,   200, 255),
    CONFIRMING:        (0,   255, 150),
    APPROACHING:       (0,   200, 0),
    SCANNING_OBSTACLE: (0,   140, 255),
    AVOIDING:          (0,   80,  255),
    REACQUIRING:       (200, 100, 0),
    ARRIVED:           (0,   255, 0),
}

# LED bar color codes per Yahboom Ctrl_WQ2812_ALL:
#   0=red  1=green  2=blue  3=yellow  4=purple  5=cyan  6=white
# A value of None means "leave LEDs in their current managed state"
# (used by ARRIVED so the dance can flash without being overridden each frame).
STATE_LED_COLORS = {
    SEARCHING:         5,   # cyan   -- scanning
    TRACKING:          3,   # yellow -- locked, centering
    CONFIRMING:        6,   # white  -- waiting on user
    APPROACHING:       1,   # green  -- driving toward target
    SCANNING_OBSTACLE: 3,   # yellow -- caution, looking around
    AVOIDING:          0,   # red    -- maneuvering around obstacle
    REACQUIRING:       4,   # purple -- hunting for lost target
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


def run_color_selection(cap):
    """
    Runs the on-screen color selection loop.
    Blocks until the user presses 1-8 or Q.
    Returns the selected key (int) or None if quit.
    """
    print('Color selection active -- press 1-8 on the camera window.')
    while True:
        ret, frame = cap.read()
        if not ret:
            return None
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
      6. Skip contours whose center is near a live blacklist entry.
      7. Return the largest surviving contour as a target dict.

    WHY NO SHAPE FILTER?
      Relying solely on color and size is more permissive -- it handles
      irregular, partially-occluded, or non-rectangular targets. The Y/N
      confirmation step is the quality gate that replaces shape filtering.

    Args:
        frame      -- BGR image (not modified)
        color_key  -- 1-8, indexes into COLORS
        blacklist  -- list of (cx, cy, expiry_time) from tracker

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

    # Size filter
    valid = [c for c in contours if cv2.contourArea(c) >= MIN_CONTOUR_AREA]

    # Blacklist filter -- skip contours too close to a rejected region
    now             = time.time()
    active_entries  = [(bx, by) for bx, by, exp in blacklist if exp > now]

    def is_blacklisted(contour):
        bx_c, by_c, bw, bh = cv2.boundingRect(contour)
        cx = bx_c + bw // 2
        cy = by_c + bh // 2
        return any(math.hypot(cx - bx, cy - by) < BLACKLIST_RADIUS_PX
                   for bx, by in active_entries)

    valid = [c for c in valid if not is_blacklisted(c)]

    if not valid:
        return None

    best       = max(valid, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(best)
    cx, cy     = x + w // 2, y + h // 2

    return {
        'x': x, 'y': y, 'w': w, 'h': h,
        'cx': cx, 'cy': cy,
        'offset_x':   cx - FRAME_CX,
        'offset_y':   cy - FRAME_CY,
        'area':       cv2.contourArea(best),
        'area_ratio': cv2.contourArea(best) / FRAME_AREA,
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
        print(f'[DR] Rotated {deg:+.1f}deg  heading now {self.heading:+.1f}deg')

    def move_forward(self, duration):
        dist = CM_PER_SECOND_FORWARD * duration
        rad  = math.radians(self.heading)
        self.x += dist * math.sin(rad)
        self.y += dist * math.cos(rad)
        print(f'[DR] Forward {dist:.1f}cm  pos ({self.x:.1f}, {self.y:.1f})')

    def strafe(self, direction, duration):
        """direction: +1 = right, -1 = left"""
        dist = CM_PER_SECOND_STRAFE * duration * direction
        rad  = math.radians(self.heading + 90)
        self.x += dist * math.sin(rad)
        self.y += dist * math.cos(rad)
        print(f'[DR] Strafe {dist:+.1f}cm  pos ({self.x:.1f}, {self.y:.1f})')

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

        # Obstacle scanning state
        self.scan_phase      = 0
        self.scan_left_dist  = 999.0
        self.scan_right_dist = 999.0

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
        """
        try:
            dist_H = self.robot.read_data_array(0x1b, 1)[0]
            dist_L = self.robot.read_data_array(0x1a, 1)[0]
            return (dist_H << 8 | dist_L) / 10.0
        except Exception as e:
            print(f'[SONAR] Read error: {e}')
            return 999.0

    def _read_sonar_avg(self, samples=SCAN_SAMPLES):
        """Average several sonar reads to reject single-frame noise."""
        readings = []
        for _ in range(samples):
            readings.append(self._read_sonar())
            time.sleep(0.02)
        return sum(readings) / len(readings) if readings else 999.0

    def _rotate_blocking(self, direction, degrees):
        """
        Rotate the body in place by an exact angle, blocking until complete.
        Used during obstacle scanning where we need precise heading changes
        and can't return to the main loop mid-rotation.
        direction: +1 = clockwise (right), -1 = counterclockwise (left)
        """
        duration = degrees / DEG_PER_SECOND_ROTATE
        speed    = SEARCH_ROTATE_SPEED
        self._drive( speed * direction,  speed * direction,
                    -speed * direction, -speed * direction)
        time.sleep(duration)
        self._stop()
        self.dr.rotate(direction, duration)

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
            print(f'[LED] error setting color: {e}')

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
            # Reset PID whenever we (re)enter TRACKING or APPROACHING so old
            # integral/derivative state from a previous run doesn't kick the
            # bot sideways on the first frame.
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
            self.blacklist.append((target['cx'], target['cy'], expiry))
            print(f'[BLACKLIST] ({target["cx"]}, {target["cy"]}) '
                  f'blocked for {BLACKLIST_DURATION:.0f}s')
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
                print(f'[REJECT-ARRIVAL] Sonar {dist:.1f}cm but target '
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
            self.scan_phase = 0
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
    # SCANNING_OBSTACLE
    # Sonar pair is fixed forward, so we rotate the *body* to point them at
    # the left side and then the right side. After both readings the bot
    # rotates back to its original heading and picks the side with greater
    # clearance, which becomes self.avoid_direction for AVOIDING.
    #
    # Phases:
    #   0  rotate 90 deg LEFT, settle, sample sonar -> scan_left_dist
    #   1  rotate 180 deg RIGHT, settle, sample sonar -> scan_right_dist
    #   2  rotate 90 deg LEFT (back to original heading), choose side, AVOID
    # -----------------------------------------------------------------------

    def _run_scanning_obstacle(self, target):
        if self.scan_phase == 0:
            self._rotate_blocking(-1, SCAN_ROTATE_ANGLE_DEG)
            time.sleep(SCAN_SETTLE_TIME)
            self.scan_left_dist = self._read_sonar_avg()
            print(f'[SCAN] Left clearance:  {self.scan_left_dist:.1f}cm')
            self.scan_phase = 1

        elif self.scan_phase == 1:
            self._rotate_blocking(+1, 2 * SCAN_ROTATE_ANGLE_DEG)
            time.sleep(SCAN_SETTLE_TIME)
            self.scan_right_dist = self._read_sonar_avg()
            print(f'[SCAN] Right clearance: {self.scan_right_dist:.1f}cm')
            self.scan_phase = 2

        elif self.scan_phase == 2:
            # Return to original heading
            self._rotate_blocking(-1, SCAN_ROTATE_ANGLE_DEG)

            # Pick the clearer side. Tie -> default to right.
            if self.scan_left_dist > self.scan_right_dist:
                self.avoid_direction = -1
                print(f'[SCAN] Going LEFT  ({self.scan_left_dist:.0f} vs '
                      f'{self.scan_right_dist:.0f})')
            else:
                self.avoid_direction = +1
                print(f'[SCAN] Going RIGHT ({self.scan_right_dist:.0f} vs '
                      f'{self.scan_left_dist:.0f})')

            self.avoid_phase      = 0
            self.last_action_time = time.time()
            self._set_state(AVOIDING)

    # -----------------------------------------------------------------------
    # AVOIDING
    # Three-phase Mecanum maneuver, parameterized by self.avoid_direction:
    #   Phase 0: strafe sideways (clear obstacle laterally)
    #   Phase 1: drive forward   (pass the obstacle)
    #   Phase 2: strafe back     (return to original lane)
    # Dead reckoning logs all movements for REACQUIRING.
    # -----------------------------------------------------------------------

    def _run_avoiding(self, target):
        # If block reappears mid-maneuver (after clearing the obstacle side),
        # skip the remaining phases and go straight to TRACKING.
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
                self.avoid_phase      = 2
                self.last_action_time = time.time()

        elif self.avoid_phase == 2:
            # Strafe back to original lane -- opposite direction of phase 0
            if elapsed < AVOID_SIDE_DURATION:
                s = AVOID_SIDE_SPEED
                self._drive( s * d, -s * d,
                            -s * d,  s * d)
            else:
                self.dr.strafe(-d, AVOID_SIDE_DURATION)
                self._stop()
                self._set_state(REACQUIRING)

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
        print(f'[REACQUIRE] Estimated bearing: {bearing:+.1f}deg')

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
        Build the dance step list once. Each entry is (drive_left, drive_right, duration).
        Three sets of [+20 right, -40 left, +20 right] = swing right, swing through
        to left, swing back to center.
        """
        s = SEARCH_ROTATE_SPEED
        t20 = 20.0 / DEG_PER_SECOND_ROTATE      # time to rotate 20 deg
        t40 = 40.0 / DEG_PER_SECOND_ROTATE      # time to rotate 40 deg
        pause = 0.08
        right = ( s,  s, -s, -s)                # rotate clockwise
        left  = (-s, -s,  s,  s)                # rotate counterclockwise
        stop  = ( 0,  0,  0,  0)
        one_set = [
            (right, t20),   # 0  -> +20
            (stop,  pause),
            (left,  t40),   # +20 -> -20
            (stop,  pause),
            (right, t20),   # -20 -> 0
            (stop,  pause),
        ]
        return one_set * 3

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

        # LED slow flash (~1 Hz)
        if time.time() - self.dance_led_last >= 0.5:
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
        elif self.state == REACQUIRING:       self._run_reacquiring(target)
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
    if tracker.state in (AVOIDING, REACQUIRING, SCANNING_OBSTACLE):
        dr = tracker.dr
        cv2.putText(frame,
                    f'DR  pos:({dr.x:.0f},{dr.y:.0f})cm  hdg:{dr.heading:.0f}deg',
                    (10, 85), font, 0.55, (200, 100, 0), 2)

    cv2.putText(frame, 'Q: quit',
                (FRAME_WIDTH - 80, 30), font, 0.5, (200, 200, 200), 1)
    return frame


# ===========================================================================
# ENTRY POINT
# ===========================================================================

def main():
    robot = Raspbot()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          TARGET_FPS)

    if not cap.isOpened():
        print('ERROR: Could not open camera.')
        del robot
        return

    # --- Color selection (blocks until user picks 1-8 or presses Q) ---
    color_key = run_color_selection(cap)
    if color_key is None:
        cap.release()
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

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        target = tracker.update(frame)

        # Keep confirming_target fresh while we're waiting for Y/N.
        # Once the state leaves CONFIRMING (by any path), clear it.
        if tracker.state == CONFIRMING:
            if target:
                confirming_target = target
        else:
            confirming_target = None

        frame = draw_overlay(frame, target, tracker)

        fps_counter += 1
        if time.time() - fps_timer >= 1.0:
            fps_display = fps_counter
            fps_counter = 0
            fps_timer   = time.time()
        cv2.putText(frame, f'FPS: {fps_display}',
                    (FRAME_WIDTH - 100, FRAME_HEIGHT - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow('Color Block Tracker', frame)

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
    cap.release()
    cv2.destroyAllWindows()
    del robot
    print('Tracker closed.')


if __name__ == '__main__':
    main()