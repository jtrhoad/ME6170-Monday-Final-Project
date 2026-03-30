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

terminal_print("=== Register Scan ===")
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
terminal_print("Scanning registers 0x00 to 0x1F continuously...")
terminal_print("Move the bot over surfaces and watch for changing values.")
terminal_print("Press Q or Ctrl+C to stop and reset.")
terminal_print("-" * 50)

# ── Keypress detection (non-blocking) ─────────────────────────────────────────
quit_flag = threading.Event()

def watch_for_q():
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

# ── Main scan loop ─────────────────────────────────────────────────────────────
try:
    while not quit_flag.is_set():
        terminal_print("")
        for reg in range(0x20):
            if quit_flag.is_set():
                break
            try:
                val = bot.read_data_array(reg, 1)[0]
                terminal_print(f"  Register 0x{reg:02x} = {val} (binary: {val:08b})")
            except Exception as e:
                terminal_print(f"  Register 0x{reg:02x} = ERROR: {e}")
        terminal_print("-" * 50)
        time.sleep(1)

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