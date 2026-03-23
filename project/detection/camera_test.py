#!/usr/bin/env python3
"""
camera_stream.py
Positions the camera to center/forward and streams live video
at 30fps locally on the Pi. Press 'q' to quit.
"""

import cv2
import time
from Raspbot_Lib import Raspbot

# --- Constants ---
SERVO_PAN  = 1          # Servo 1 controls left/right
SERVO_TILT = 2          # Servo 2 controls up/down
PAN_CENTER  = 72        # 0-180 degrees, 90 = straight ahead
TILT_CENTER = 25        # 0-100 degrees (Raspbot_Lib caps servo 2 at 100)
                        # 50 = roughly level/forward

CAMERA_INDEX  = 0       # USB camera
FRAME_WIDTH   = 640
FRAME_HEIGHT  = 480
TARGET_FPS    = 60


def center_camera(robot):
    """
    Drive to center from consistent directions to eliminate
    servo hysteresis. 
    Pan  - always approach from 180 â†’ lands at 72
    Tilt - always approach from 0   â†’ lands at 25
    """
    # Pan: drive high first, then come down to center
    robot.Ctrl_Servo(SERVO_PAN, 180)
    time.sleep(0.5)
    robot.Ctrl_Servo(SERVO_PAN, PAN_CENTER)
    time.sleep(0.5)

    # Tilt: drive low first, then come up to center
    robot.Ctrl_Servo(SERVO_TILT, 0)
    time.sleep(0.5)
    robot.Ctrl_Servo(SERVO_TILT, TILT_CENTER)
    time.sleep(0.5)

    print(f'Camera centered: pan={PAN_CENTER}, tilt={TILT_CENTER}')


def main():
    # Initialize robot and center camera
    robot = Raspbot()
    center_camera(robot)

    # Open USB camera
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          TARGET_FPS)
    # MJPG codec gives smoother USB camera throughput
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M','J','P','G'))

    if not cap.isOpened():
        print('ERROR: Could not open camera.')
        return

    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f'Camera opened at {FRAME_WIDTH}x{FRAME_HEIGHT} @ {actual_fps}fps')
    print('Streaming... press q to quit.')

    while True:
        ret, frame = cap.read()

        if not ret:
            print('WARNING: Failed to grab frame.')
            break

        cv2.imshow('Raspbot Camera Stream', frame)

        # Wait 1ms between frames, exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print('Stream closed.')


if __name__ == '__main__':
    main()