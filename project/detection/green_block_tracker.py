#!/usr/bin/env python3
"""
green_block_tracker.py
======================
Full state machine for finding, tracking, and approaching a green block
while avoiding obstacles. Designed for the Raspbot V2 with Mecanum wheels.

DEPLOYMENT: Run this on the Raspbot Pi. Requires:
  - Raspbot_Lib    (pre-installed in Yahboom image)
  - opencv-python  (pre-installed in Yahboom image)
  - numpy          (pre-installed in Yahboom image)

Before running, calibrate DEG_PER_SECOND_ROTATE, CM_PER_SECOND_FORWARD,
and CM_PER_SECOND_STRAFE using dr_calibration.py on the actual bot surface.

STATE MACHINE:
  SEARCHING   -- No block visible. Rotate bot in place to scan.
  TRACKING    -- Block found. Pan/tilt camera to center it. Stop wheels.
  APPROACHING -- Camera centered on block. Drive forward toward it.
  AVOIDING    -- Obstacle within threshold. Navigate around it using
                 dead reckoning to preserve estimated block position.
  REACQUIRING -- Obstacle cleared. Use estimated block angle to rotate
                 back toward where the block should be.

DEAD RECKONING:
  During AVOIDING we can't see the block, so we track all movements made
  (rotation angle, distance forward/back) and compute an estimated bearing
  to the block's last known position. This gives REACQUIRING a starting
  angle to rotate toward instead of doing a full blind search again.

CONTROLS:
  Q  ->  quit
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
TARGET_FPS   = 30

# --- Servo ---
SERVO_PAN         = 1
SERVO_TILT        = 2
PAN_CENTER        = 72
TILT_CENTER       = 25
PAN_MIN,  PAN_MAX  =  5, 175
TILT_MIN, TILT_MAX =  5,  95

# How many pixels of offset before we move the servo (deadband)
# Prevents jitter when the block is nearly centered
SERVO_DEADBAND    = 15
SERVO_P_GAIN      = 0.05      # offset_px * gain = servo degrees to move

# --- Green HSV ---
GREEN_LOWER = np.array([40,  60,  40])
GREEN_UPPER = np.array([80,  255, 255])

# --- Shape Filter ---
MIN_CONTOUR_AREA = 1500
SHAPE_VERTEX_MIN = 4
SHAPE_VERTEX_MAX = 6
MIN_SOLIDITY     = 0.80
MIN_ASPECT_RATIO = 0.30
MAX_ASPECT_RATIO = 3.50

# --- Approach ---
# Stop when block fills this fraction of the frame (proxy for distance).
# On the real bot, supplement with ultrasonic for accuracy.
APPROACH_AREA_STOP = 0.25    # 25% of frame area = close enough
APPROACH_SPEED     = 50

# --- Obstacle Avoidance ---
OBSTACLE_DISTANCE_CM   = 30.0
AVOID_SIDE_SPEED       = 60
AVOID_SIDE_DURATION    = 0.8
AVOID_FORWARD_DURATION = 1.0
AVOID_SPEED            = 50

# --- Search ---
SEARCH_ROTATE_SPEED  = 40
SEARCH_ROTATE_STEP   = 0.3   # seconds per rotation step before rechecking frame
SEARCH_MAX_STEPS     = 24    # ~360 degrees total

# --- Dead Reckoning calibration (tune on actual bot surface) ---
DEG_PER_SECOND_ROTATE = 50.0
CM_PER_SECOND_FORWARD = 20.0
CM_PER_SECOND_STRAFE  = 18.0

# ===========================================================================
# STATES
# ===========================================================================

SEARCHING   = 'SEARCHING'
TRACKING    = 'TRACKING'
APPROACHING = 'APPROACHING'
AVOIDING    = 'AVOIDING'
REACQUIRING = 'REACQUIRING'

FRAME_CX   = FRAME_WIDTH  // 2
FRAME_CY   = FRAME_HEIGHT // 2
FRAME_AREA = FRAME_WIDTH * FRAME_HEIGHT


# ===========================================================================
# SHAPE DETECTION
# ===========================================================================

def is_rectangle(contour):
    """
    Returns True if a contour passes all three rectangle gates:
      1. Vertex count  -- approxPolyDP simplifies to key corners (4-6)
      2. Solidity      -- area / hull area ratio (>0.80 = compact solid shape)
      3. Aspect ratio  -- width / height (filters lines and flat strips)
    """
    perimeter = cv2.arcLength(contour, True)
    approx    = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
    if not (SHAPE_VERTEX_MIN <= len(approx) <= SHAPE_VERTEX_MAX):
        return False

    hull = cv2.convexHull(contour)
    if cv2.contourArea(hull) == 0:
        return False
    if cv2.contourArea(contour) / cv2.contourArea(hull) < MIN_SOLIDITY:
        return False

    _, _, w, h = cv2.boundingRect(contour)
    if h == 0:
        return False
    return MIN_ASPECT_RATIO <= (w / h) <= MAX_ASPECT_RATIO


def find_green_block(frame):
    """
    Find the largest green rectangle in the frame.
    Returns a target dict or None.
    """
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    valid = [c for c in contours
             if cv2.contourArea(c) >= MIN_CONTOUR_AREA and is_rectangle(c)]
    if not valid:
        return None

    best        = max(valid, key=cv2.contourArea)
    x, y, w, h  = cv2.boundingRect(best)
    cx, cy      = x + w // 2, y + h // 2

    return {
        'x': x, 'y': y, 'w': w, 'h': h,
        'cx': cx, 'cy': cy,
        'offset_x':   cx - FRAME_CX,
        'offset_y':   cy - FRAME_CY,
        'area':       cv2.contourArea(best),
        'area_ratio': cv2.contourArea(best) / FRAME_AREA
    }


# ===========================================================================
# DEAD RECKONING
# ===========================================================================

class DeadReckoning:
    """
    Estimates position and heading relative to where the block was last seen.

    Coordinate system (relative to last sighting):
      heading : degrees, 0 = facing block, + = rotated right
      x       : cm, + = right,   - = left
      y       : cm, + = forward, - = backward

    After avoiding an obstacle, estimated_block_bearing gives the angle
    to rotate toward in order to face where the block should still be.

    WHY DEAD RECKONING INSTEAD OF FULL SLAM/LOCALIZATION?
      Dead reckoning uses known commands (speed x time) to estimate position.
      It accumulates error but is simple and runs on the Pi with no extra
      hardware. For short obstacle avoidance maneuvers it's accurate enough.
      If you later add an IMU or encoders, those readings can replace the
      time-based estimates here for much better accuracy.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.heading = 0.0
        self.x       = 0.0
        self.y       = 0.0

    def rotate(self, direction, duration):
        """direction: +1 = clockwise (right), -1 = counterclockwise"""
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
        Angle (degrees) to rotate to face the block's last known position
        from our current estimated position and heading.
        """
        bearing = math.degrees(math.atan2(-self.x, -self.y))
        return bearing - self.heading


# ===========================================================================
# TRACKER STATE MACHINE
# ===========================================================================

class GreenBlockTracker:

    def __init__(self, robot):
        self.robot       = robot
        self.state       = SEARCHING
        self.dr          = DeadReckoning()
        self.pan         = PAN_CENTER
        self.tilt        = TILT_CENTER
        self.search_steps     = 0
        self.avoid_phase      = 0
        self.last_action_time = time.time()
        self._center_camera()

    def _center_camera(self):
        self.robot.Ctrl_Servo(SERVO_PAN,  PAN_CENTER)
        self.robot.Ctrl_Servo(SERVO_TILT, TILT_CENTER)
        self.pan  = PAN_CENTER
        self.tilt = TILT_CENTER

    def _set_state(self, new_state):
        if new_state != self.state:
            print(f'[STATE] {self.state} -> {new_state}')
            self.state            = new_state
            self.last_action_time = time.time()

    def _elapsed(self):
        return time.time() - self.last_action_time

    # --- Proportional servo controller ---

    def _update_servos(self, target):
        """
        Moves each servo proportionally to the pixel offset from center.
        WHY PROPORTIONAL?
          As the servo moves toward center, the offset shrinks naturally,
          so the correction automatically slows down -- no overshoot tuning
          needed for a simple P controller at this stage.
        """
        if abs(target['offset_x']) > SERVO_DEADBAND:
            delta    = int(target['offset_x'] * SERVO_P_GAIN)
            self.pan = int(np.clip(self.pan + delta, PAN_MIN, PAN_MAX))
            self.robot.Ctrl_Servo(SERVO_PAN, self.pan)

        if abs(target['offset_y']) > SERVO_DEADBAND:
            delta     = int(target['offset_y'] * SERVO_P_GAIN)
            self.tilt = int(np.clip(self.tilt + delta, TILT_MIN, TILT_MAX))
            self.robot.Ctrl_Servo(SERVO_TILT, self.tilt)

    def _is_centered(self, target):
        return (abs(target['offset_x']) <= SERVO_DEADBAND and
                abs(target['offset_y']) <= SERVO_DEADBAND)

    # -----------------------------------------------------------------------
    # SEARCHING
    # Rotate bot in place. Camera stays centered -- wheels find the block
    # first, then camera fine-tunes once block is visible.
    # -----------------------------------------------------------------------

    def _run_searching(self, target):
        if target:
            self.robot.Car_Stop()
            self.search_steps = 0
            self.dr.reset()
            self._set_state(TRACKING)
            return

        if self._elapsed() >= SEARCH_ROTATE_STEP:
            # In-place clockwise rotation: left wheels forward, right backward
            self.robot.Ctrl_Car( SEARCH_ROTATE_SPEED, -SEARCH_ROTATE_SPEED,
                                 SEARCH_ROTATE_SPEED, -SEARCH_ROTATE_SPEED)
            self.dr.rotate(+1, SEARCH_ROTATE_STEP)
            self.search_steps        += 1
            self.last_action_time     = time.time()

            if self.search_steps >= SEARCH_MAX_STEPS:
                # Full 360 done -- nudge forward then restart scan
                print('[SEARCH] Full rotation complete, advancing.')
                self.robot.Car_Stop()
                self.robot.Ctrl_Car(APPROACH_SPEED, APPROACH_SPEED,
                                    APPROACH_SPEED, APPROACH_SPEED)
                time.sleep(0.5)
                self.dr.move_forward(0.5)
                self.robot.Car_Stop()
                self.search_steps = 0
                self.dr.reset()

    # -----------------------------------------------------------------------
    # TRACKING
    # Block is visible. Pan/tilt camera to center it. Wheels stopped.
    # Once centered, transition to APPROACHING.
    # -----------------------------------------------------------------------

    def _run_tracking(self, target):
        if not target:
            self.robot.Car_Stop()
            self._set_state(SEARCHING)
            return

        self._update_servos(target)

        if self._is_centered(target):
            self._set_state(APPROACHING)

    # -----------------------------------------------------------------------
    # APPROACHING
    # Drive forward while keeping camera centered on block.
    # Stop if obstacle detected or if we're close enough (area threshold).
    # -----------------------------------------------------------------------

    def _run_approaching(self, target):
        if not target:
            self.robot.Car_Stop()
            self._set_state(SEARCHING)
            return

        self._update_servos(target)

        # Obstacle check
        dist = self.robot.Get_Sonar()
        if dist < OBSTACLE_DISTANCE_CM:
            self.robot.Car_Stop()
            print(f'[AVOID] Obstacle at {dist:.1f}cm')
            self.avoid_phase = 0
            self._set_state(AVOIDING)
            return

        # Arrival check
        if target['area_ratio'] >= APPROACH_AREA_STOP:
            self.robot.Car_Stop()
            print('[DONE] Reached green block.')
            return

        self.robot.Ctrl_Car(APPROACH_SPEED, APPROACH_SPEED,
                            APPROACH_SPEED, APPROACH_SPEED)

    # -----------------------------------------------------------------------
    # AVOIDING
    # Can't see block (or obstacle blocking path). Three-phase maneuver:
    #   Phase 0: strafe right
    #   Phase 1: drive forward past obstacle
    #   Phase 2: strafe left back into lane
    # Dead reckoning logs all movements for REACQUIRING.
    # -----------------------------------------------------------------------

    def _run_avoiding(self, target):
        # If block reappears after clearing, grab it immediately
        if target and self.avoid_phase > 0:
            self.robot.Car_Stop()
            self._set_state(TRACKING)
            return

        elapsed = self._elapsed()

        if self.avoid_phase == 0:
            # Strafe right -- Mecanum: FL back, FR forward, RL forward, RR back
            if elapsed < AVOID_SIDE_DURATION:
                self.robot.Ctrl_Car( AVOID_SIDE_SPEED, -AVOID_SIDE_SPEED,
                                    -AVOID_SIDE_SPEED,  AVOID_SIDE_SPEED)
            else:
                self.dr.strafe(+1, AVOID_SIDE_DURATION)
                self.robot.Car_Stop()
                self.avoid_phase      = 1
                self.last_action_time = time.time()

        elif self.avoid_phase == 1:
            # Move forward past the obstacle
            if elapsed < AVOID_FORWARD_DURATION:
                self.robot.Ctrl_Car(AVOID_SPEED, AVOID_SPEED,
                                    AVOID_SPEED, AVOID_SPEED)
            else:
                self.dr.move_forward(AVOID_FORWARD_DURATION)
                self.robot.Car_Stop()
                self.avoid_phase      = 2
                self.last_action_time = time.time()

        elif self.avoid_phase == 2:
            # Strafe left back into original lane
            if elapsed < AVOID_SIDE_DURATION:
                self.robot.Ctrl_Car(-AVOID_SIDE_SPEED,  AVOID_SIDE_SPEED,
                                     AVOID_SIDE_SPEED, -AVOID_SIDE_SPEED)
            else:
                self.dr.strafe(-1, AVOID_SIDE_DURATION)
                self.robot.Car_Stop()
                self._set_state(REACQUIRING)

    # -----------------------------------------------------------------------
    # REACQUIRING
    # Obstacle is behind us. Use dead reckoning bearing to rotate toward
    # where the block should still be, then hand off to SEARCHING/TRACKING.
    # -----------------------------------------------------------------------

    def _run_reacquiring(self, target):
        if target:
            self.robot.Car_Stop()
            self._set_state(TRACKING)
            return

        bearing = self.dr.estimated_block_bearing
        print(f'[REACQUIRE] Estimated bearing to block: {bearing:+.1f}deg')

        if abs(bearing) > 10:
            direction   = +1 if bearing > 0 else -1
            rotate_time = min(abs(bearing) / DEG_PER_SECOND_ROTATE, 0.5)
            self.robot.Ctrl_Car(
                 SEARCH_ROTATE_SPEED * direction, -SEARCH_ROTATE_SPEED * direction,
                 SEARCH_ROTATE_SPEED * direction, -SEARCH_ROTATE_SPEED * direction
            )
            time.sleep(rotate_time)
            self.dr.rotate(direction, rotate_time)
            self.robot.Car_Stop()
        else:
            # Close enough -- hand off to SEARCHING for fine scan
            self.dr.reset()
            self._set_state(SEARCHING)

    # -----------------------------------------------------------------------
    # MAIN UPDATE (called once per frame)
    # -----------------------------------------------------------------------

    def update(self, frame):
        target = find_green_block(frame)

        if   self.state == SEARCHING:   self._run_searching(target)
        elif self.state == TRACKING:    self._run_tracking(target)
        elif self.state == APPROACHING: self._run_approaching(target)
        elif self.state == AVOIDING:    self._run_avoiding(target)
        elif self.state == REACQUIRING: self._run_reacquiring(target)

        return target


# ===========================================================================
# DISPLAY
# ===========================================================================

STATE_COLORS = {
    SEARCHING:   (100, 100, 100),
    TRACKING:    (0,   200, 255),
    APPROACHING: (0,   200, 0),
    AVOIDING:    (0,   80,  255),
    REACQUIRING: (200, 100, 0),
}

def draw_overlay(frame, target, tracker):
    color = STATE_COLORS.get(tracker.state, (255, 255, 255))

    # Frame center crosshair
    cv2.drawMarker(frame, (FRAME_CX, FRAME_CY),
                   (255, 255, 255), cv2.MARKER_CROSS, 20, 2)

    if target:
        cv2.rectangle(frame,
                      (target['x'], target['y']),
                      (target['x'] + target['w'], target['y'] + target['h']),
                      (0, 200, 0), 2)
        cv2.putText(frame, 'GREEN BLOCK',
                    (target['x'], target['y'] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 0), 2)
        cv2.circle(frame, (target['cx'], target['cy']), 6, (0, 255, 255), -1)
        cv2.line(frame, (FRAME_CX, FRAME_CY),
                 (target['cx'], target['cy']), (255, 255, 0), 2)

        ox, oy = target['offset_x'], target['offset_y']
        cv2.putText(frame, f'Offset  X:{ox:+d}  Y:{oy:+d}',
                    (10, FRAME_HEIGHT - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f'Area: {target["area_ratio"]*100:.1f}% of frame',
                    (10, FRAME_HEIGHT - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    else:
        cv2.putText(frame, 'No target',
                    (10, FRAME_HEIGHT - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 80), 2)

    # State + FPS
    cv2.putText(frame, f'STATE: {tracker.state}',
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Dead reckoning readout during AVOIDING / REACQUIRING
    if tracker.state in (AVOIDING, REACQUIRING):
        dr = tracker.dr
        cv2.putText(frame,
                    f'DR  pos:({dr.x:.0f},{dr.y:.0f})cm  hdg:{dr.heading:.0f}deg',
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 100, 0), 2)

    cv2.putText(frame, 'Q: quit',
                (FRAME_WIDTH - 80, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    return frame


# ===========================================================================
# ENTRY POINT
# ===========================================================================

def main():
    robot   = Raspbot()
    tracker = GreenBlockTracker(robot)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          TARGET_FPS)

    if not cap.isOpened():
        print(f'ERROR: Could not open camera at index {CAMERA_INDEX}.')
        del robot
        return

    print('Green Block Tracker running.')
    print('SEARCHING -> TRACKING -> APPROACHING')
    print('AVOIDING (obstacle) -> REACQUIRING (dead reckoning)')
    print('Press Q to quit.')

    fps_counter = 0
    fps_display = 0
    fps_timer   = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        target = tracker.update(frame)
        frame  = draw_overlay(frame, target, tracker)

        fps_counter += 1
        if time.time() - fps_timer >= 1.0:
            fps_display = fps_counter
            fps_counter = 0
            fps_timer   = time.time()

        cv2.putText(frame, f'FPS: {fps_display}',
                    (FRAME_WIDTH - 100, FRAME_HEIGHT - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow('Green Block Tracker', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    robot.Car_Stop()
    cap.release()
    cv2.destroyAllWindows()
    del robot
    print('Tracker closed.')


if __name__ == '__main__':
    main()
