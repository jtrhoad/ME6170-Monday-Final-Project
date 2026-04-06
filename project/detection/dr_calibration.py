#!/usr/bin/env python3
"""
dr_calibration.py
=================
Dead reckoning calibration tests for the Raspbot V2.
Run this on the actual bot surface before using green_block_tracker.py.

WHAT THIS DOES:
  Runs three timed movement tests -- rotation, forward, and strafe.
  You measure the actual distance/angle with a tape measure or protractor,
  then divide by the test duration to get the calibration constants.

CALIBRATION CONSTANTS TO UPDATE in green_block_tracker.py:
  DEG_PER_SECOND_ROTATE  -- from Test 1
  CM_PER_SECOND_FORWARD  -- from Test 2
  CM_PER_SECOND_STRAFE   -- from Test 3

USAGE:
  python dr_calibration.py

SAFETY:
  - Place the bot in a clear area with at least 1m space in all directions.
  - Each test runs for a fixed duration then stops automatically.
  - Press Ctrl+C at any time to emergency stop.
"""

import time
import sys
from Raspbot_Lib import Raspbot

# ===========================================================================
# TEST PARAMETERS -- match these to the values in green_block_tracker.py
# ===========================================================================

SEARCH_ROTATE_SPEED  = 40    # Must match green_block_tracker.py
APPROACH_SPEED       = 50    # Must match green_block_tracker.py
AVOID_SIDE_SPEED     = 60    # Must match green_block_tracker.py

TEST_DURATION        = 2.0   # Seconds per test -- longer = easier to measure accurately

# ===========================================================================
# HELPERS
# ===========================================================================

def countdown(seconds, message):
    """Give the user time to position themselves before a test runs."""
    print(f'\n{message}')
    for i in range(seconds, 0, -1):
        print(f'  Starting in {i}...')
        time.sleep(1)
    print('  GO!\n')


def wait_for_enter(prompt='Press ENTER to continue...'):
    input(f'\n{prompt}')


def print_result(test_name, duration, measured, unit, constant_name):
    """Print the measured result and the constant to plug in."""
    rate = measured / duration
    print(f'\n{"="*55}')
    print(f'  {test_name} RESULT')
    print(f'{"="*55}')
    print(f'  Test duration : {duration:.1f} seconds')
    print(f'  You measured  : {measured:.1f} {unit}')
    print(f'  Calculated    : {measured:.1f} / {duration:.1f} = {rate:.2f} {unit}/sec')
    print(f'\n  Update in green_block_tracker.py:')
    print(f'    {constant_name} = {rate:.2f}')
    print(f'{"="*55}')


# ===========================================================================
# TEST 1: ROTATION
# Spins the bot clockwise in place for TEST_DURATION seconds.
# Measure the total angle rotated with a protractor or by marking the floor.
# ===========================================================================

def test_rotation(robot):
    print('\n' + '='*55)
    print('  TEST 1: IN-PLACE ROTATION')
    print('='*55)
    print(f'  The bot will rotate clockwise for {TEST_DURATION:.1f} seconds.')
    print('  HOW TO MEASURE:')
    print('    1. Place a piece of tape on the floor under the front of the bot.')
    print('    2. Mark the starting direction with a second piece of tape.')
    print('    3. After the test, measure the angle between the two marks.')
    print('    4. Enter that angle in degrees when prompted.')

    wait_for_enter('Position the bot and press ENTER when ready...')
    countdown(3, 'Rotation test starting...')

    try:
        # Mecanum in-place clockwise: left wheels forward, right wheels backward
        robot.Ctrl_Car( SEARCH_ROTATE_SPEED, -SEARCH_ROTATE_SPEED,
                        SEARCH_ROTATE_SPEED, -SEARCH_ROTATE_SPEED)
        time.sleep(TEST_DURATION)
    finally:
        robot.Car_Stop()

    print(f'\nBot stopped. Measure the angle rotated.')
    while True:
        try:
            measured = float(input('  Enter measured angle (degrees): '))
            break
        except ValueError:
            print('  Please enter a number.')

    print_result('ROTATION', TEST_DURATION, measured,
                 'degrees', 'DEG_PER_SECOND_ROTATE')
    return measured / TEST_DURATION


# ===========================================================================
# TEST 2: FORWARD MOVEMENT
# Drives the bot straight forward for TEST_DURATION seconds.
# Measure the distance traveled with a tape measure.
# ===========================================================================

def test_forward(robot):
    print('\n' + '='*55)
    print('  TEST 2: FORWARD MOVEMENT')
    print('='*55)
    print(f'  The bot will drive forward for {TEST_DURATION:.1f} seconds.')
    print('  HOW TO MEASURE:')
    print('    1. Place a mark at the starting position of the bot (front edge).')
    print('    2. After the test, measure from the mark to the new front edge.')
    print('    3. Enter that distance in centimeters when prompted.')

    wait_for_enter('Position the bot on a clear straight path and press ENTER...')
    countdown(3, 'Forward movement test starting...')

    try:
        robot.Ctrl_Car(APPROACH_SPEED, APPROACH_SPEED, APPROACH_SPEED)
        time.sleep(TEST_DURATION)
    finally:
        robot.Car_Stop()

    print(f'\nBot stopped. Measure the distance traveled.')
    while True:
        try:
            measured = float(input('  Enter measured distance (cm): '))
            break
        except ValueError:
            print('  Please enter a number.')

    print_result('FORWARD', TEST_DURATION, measured,
                 'cm', 'CM_PER_SECOND_FORWARD')
    return measured / TEST_DURATION


# ===========================================================================
# TEST 3: STRAFE (LATERAL MOVEMENT)
# Drives the bot sideways (right) using Mecanum wheels for TEST_DURATION secs.
# Mecanum strafing is less efficient than forward movement -- expect a lower
# rate than forward due to wheel roller losses.
# ===========================================================================

def test_strafe(robot):
    print('\n' + '='*55)
    print('  TEST 3: STRAFE (SIDEWAYS) MOVEMENT')
    print('='*55)
    print(f'  The bot will strafe RIGHT for {TEST_DURATION:.1f} seconds.')
    print('  HOW TO MEASURE:')
    print('    1. Place a mark at the left edge of the bot.')
    print('    2. After the test, measure how far right the left edge moved.')
    print('    3. Enter that distance in centimeters when prompted.')
    print()
    print('  NOTE: Mecanum strafing loses some efficiency through the rollers.')
    print('  Expect a lower cm/sec than forward movement -- this is normal.')

    wait_for_enter('Position the bot with clear space to the right and press ENTER...')
    countdown(3, 'Strafe test starting...')

    try:
        # Mecanum strafe right:
        #   FL: backward  FR: forward
        #   RL: forward   RR: backward
        robot.Ctrl_Car(-AVOID_SIDE_SPEED,  AVOID_SIDE_SPEED,
                        AVOID_SIDE_SPEED, -AVOID_SIDE_SPEED)
        time.sleep(TEST_DURATION)
    finally:
        robot.Car_Stop()

    print(f'\nBot stopped. Measure the lateral distance traveled.')
    while True:
        try:
            measured = float(input('  Enter measured distance (cm): '))
            break
        except ValueError:
            print('  Please enter a number.')

    print_result('STRAFE', TEST_DURATION, measured,
                 'cm', 'CM_PER_SECOND_STRAFE')
    return measured / TEST_DURATION


# ===========================================================================
# SUMMARY
# ===========================================================================

def print_summary(deg_per_sec, cm_per_sec_fwd, cm_per_sec_strafe):
    print('\n\n' + '='*55)
    print('  CALIBRATION SUMMARY')
    print('  Copy these values into green_block_tracker.py')
    print('='*55)
    print(f'  DEG_PER_SECOND_ROTATE = {deg_per_sec:.2f}')
    print(f'  CM_PER_SECOND_FORWARD = {cm_per_sec_fwd:.2f}')
    print(f'  CM_PER_SECOND_STRAFE  = {cm_per_sec_strafe:.2f}')
    print('='*55)
    print()
    print('  TIP: Run each test 2-3 times and average the results.')
    print('  Surface friction varies -- calibrate on the same floor')
    print('  you plan to run the full tracker on.')
    print()


# ===========================================================================
# MENU
# ===========================================================================

def main():
    print('='*55)
    print('  RASPBOT V2 -- DEAD RECKONING CALIBRATION')
    print('='*55)
    print()
    print('  This script runs three movement tests to measure the')
    print('  actual speed/rotation rate of the bot at the speeds')
    print('  used in green_block_tracker.py.')
    print()
    print('  Tests:')
    print('    1. In-place rotation  (DEG_PER_SECOND_ROTATE)')
    print('    2. Forward movement   (CM_PER_SECOND_FORWARD)')
    print('    3. Strafe movement    (CM_PER_SECOND_STRAFE)')
    print()
    print('  SAFETY: Clear at least 1m of space around the bot.')
    print('          Press Ctrl+C at any time to emergency stop.')
    print()

    robot = Raspbot()

    results = {}

    while True:
        print('\nSelect a test:')
        print('  1  Rotation test')
        print('  2  Forward movement test')
        print('  3  Strafe movement test')
        print('  4  Run all three in sequence')
        print('  Q  Quit')
        print()

        choice = input('  Enter choice: ').strip().lower()

        try:
            if choice == '1':
                results['deg'] = test_rotation(robot)
            elif choice == '2':
                results['fwd'] = test_forward(robot)
            elif choice == '3':
                results['strafe'] = test_strafe(robot)
            elif choice == '4':
                results['deg']    = test_rotation(robot)
                results['fwd']    = test_forward(robot)
                results['strafe'] = test_strafe(robot)
                print_summary(results['deg'], results['fwd'], results['strafe'])
            elif choice == 'q':
                break
            else:
                print('  Invalid choice.')

        except KeyboardInterrupt:
            robot.Car_Stop()
            print('\n\n  Emergency stop triggered.')
            print('  Bot stopped. Returning to menu.')

    # Print summary if we have all three results
    if all(k in results for k in ('deg', 'fwd', 'strafe')):
        print_summary(results['deg'], results['fwd'], results['strafe'])

    del robot
    print('Calibration session complete.')


if __name__ == '__main__':
    main()
