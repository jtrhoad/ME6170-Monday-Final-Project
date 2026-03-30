#!/usr/bin/env python3

import sys
import time

sys.path.append('/home/pi/project_demo/lib')
from McLumk_Wheel_Sports import *

# ── Speed settings ─────────────────────────────────────────────────────────────
SPEED       = 30
SPEED_SHARP = int(SPEED * 1.5)  # used for tight turns

def read_ir():
    """Read the 4-channel IR line sensor from I2C register 0x0a.
    Returns (L1, L2, R1, R2) where 0 = line detected, 1 = no line.
    Sensor layout:   L1  L2  |  R1  R2
                     far in  |  in  far
    """
    track = int(bot.read_data_array(0x0a, 1)[0])
    L1 = (track >> 3) & 0x01   # leftmost
    L2 = (track >> 2) & 0x01   # inner left
    R1 = (track >> 1) & 0x01   # inner right
    R2 =  track       & 0x01   # rightmost
    return L1, L2, R1, R2


def main():
    print("IR line following started. Press Ctrl+C to stop.")
    print("NOTE: If using blue tape, ensure good contrast with the floor surface.")

    try:
        while True:
            L1, L2, R1, R2 = read_ir()

            # ── All four sensors on the line → go straight ─────────────────
            if L1 == 0 and L2 == 0 and R1 == 0 and R2 == 0:
                move_forward(SPEED)

            # ── Both centre sensors on line → go straight ──────────────────
            elif L2 == 0 and R1 == 0:
                move_forward(SPEED)

            # ── Right acute / big right bend (left sensors + R2 see line) ──
            elif (L2 == 0 or L1 == 0) and R2 == 0:
                rotate_right(SPEED)
                time.sleep(0.05)

            # ── Left acute / big left bend (L1 + right sensors see line) ───
            elif L1 == 0 and (R2 == 0 or R1 == 0):
                rotate_left(SPEED_SHARP)
                time.sleep(0.15)

            # ── Drifting right — only outermost left sees line ──────────────
            elif L1 == 0:
                rotate_left(SPEED)
                time.sleep(0.02)

            # ── Drifting left — only outermost right sees line ──────────────
            elif R2 == 0:
                rotate_right(SPEED)
                time.sleep(0.01)

            # ── Slightly left of centre ─────────────────────────────────────
            elif L2 == 0 and R1 == 1:
                rotate_left(SPEED)

            # ── Slightly right of centre ────────────────────────────────────
            elif L2 == 1 and R1 == 0:
                rotate_right(SPEED)

            # ── No line detected at all → stop and wait ─────────────────────
            else:
                stop_robot()

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nStopped by user.")

    finally:
        stop_robot()
        del bot
        print("Done.")


if __name__ == "__main__":
    main()