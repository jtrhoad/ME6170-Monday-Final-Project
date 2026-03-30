#!/usr/bin/env python3

import sys
import time
import tty
import termios
import threading

def terminal_print(msg):
    """Print with proper carriage return for raw terminal mode."""
    sys.stdout.write(msg + "\r\n")
    sys.stdout.flush()

terminal_print("=== IR Sensor Test ===")
terminal_print("Starting up...")

try:
    sys.path.append('/home/pi/project_demo/lib')
    from McLumk_Wheel_Sports import *
    terminal_print("Library loaded OK")
except Exception as e:
    terminal_print(f"ERROR loading library: {e}")
    sys.exit(1)

try:
    _ = bot
    terminal_print("Bot connected OK")
except Exception as e:
    terminal_print(f"ERROR connecting to bot: {e}")
    sys.exit(1)

terminal_print("")
terminal_print("Reading sensors every 0.2 seconds...")
terminal_print("Move the bot over your tape and watch the values change.")
terminal_print("Press Ctrl+C or Q to stop and reset.")
terminal_print("")
terminal_print("L1(far left)  L2(inner left)  R1(inner right)  R2(far right)  RAW")
terminal_print("-" * 65)

# ── Keypress detection (non-blocking) ─────────────────────────────────────────
quit_flag = threading.Event()

def watch_for_q():
    """Runs in a background thread — sets quit_flag when Q is pressed."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        while not quit_flag.is_set():
            ch = sys.stdin.read(1)
            if ch.lower() == 'q':
                terminal_print("\r\nQ pressed — resetting bot...")
                quit_flag.set()
                break
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

key_thread = threading.Thread(target=watch_for_q, daemon=True)
key_thread.start()

# ── Main sensor loop ───────────────────────────────────────────────────────────
try:
    while not quit_flag.is_set():
        try:
            raw   = bot.read_data_array(0x0a, 1)[0]
            track = int(raw)

            L1 = (track >> 3) & 0x01
            L2 = (track >> 2) & 0x01
            R1 = (track >> 1) & 0x01
            R2 =  track       & 0x01

            bar  = "[##]" if L1 == 0 else "[  ]"
            bar += "[##]" if L2 == 0 else "[  ]"
            bar += "[##]" if R1 == 0 else "[  ]"
            bar += "[##]" if R2 == 0 else "[  ]"

            terminal_print(f"L1={L1}  L2={L2}  R1={R1}  R2={R2}  raw={track:08b} ({track})  {bar}")

        except Exception as e:
            terminal_print(f"ERROR reading sensor: {e}")

        time.sleep(0.2)

except KeyboardInterrupt:
    terminal_print("\r\nCtrl+C pressed — resetting bot...")
    quit_flag.set()

finally:
    try:
        stop_robot()
        terminal_print("Motors stopped.")
    except:
        pass
    try:
        del bot
        terminal_print("Bot object deleted and reset.")
    except:
        pass
    terminal_print("Done.")