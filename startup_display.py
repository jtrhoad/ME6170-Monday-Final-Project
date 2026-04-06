#!/usr/bin/env python3
"""
startup_display.py
------------------
Displays the robot's IP address and battery level on the Raspbot V2 OLED.
Loops every UPDATE_INTERVAL seconds so values stay current.

Hardware context:
  - OLED: SSD1306 128x64, I2C address 0x3C
  - Battery ADC: Yahboom expansion board, I2C address 0x18
  - Both share I2C bus 1 (GPIO 2 / 3 on the Raspberry Pi 5)
"""

import time
import socket
import smbus2
from luma.core.interface.serial import i2c
from luma.oled.device import ssd1306
from PIL import Image, ImageDraw, ImageFont

# ─── Constants ────────────────────────────────────────────────────────────────

I2C_BUS             = 1       # Raspberry Pi hardware I2C bus
OLED_ADDRESS        = 0x3C    # SSD1306 OLED I2C address
BATTERY_ADDRESS     = 0x2B    # Yahboom expansion board ADC address
BATTERY_REGISTER    = 0x00    # Register that holds the raw ADC reading

# 2S LiPo voltage range used by Raspbot V2
BATTERY_MAX_V       = 8.40    # Fully charged (4.20 V × 2 cells)
BATTERY_MIN_V       = 6.40    # Safe discharge cutoff (3.20 V × 2 cells)

UPDATE_INTERVAL     = 10      # Seconds between screen refreshes

# ─── Network ──────────────────────────────────────────────────────────────────

def get_ip_address() -> str:
    """
    Retrieves the active outbound IP address.

    Why this approach: connecting a UDP socket to an external address
    (without sending data) forces the OS to select the correct network
    interface. We then read back which local IP was chosen.
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(2)
        s.connect(("8.8.8.8", 80))   # Google DNS — no data is actually sent
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "No network"

# ─── Battery ──────────────────────────────────────────────────────────────────

def get_uptime() -> str:
    """
    Reads system uptime from /proc/uptime — a file the Linux kernel updates
    every second with total seconds since boot.

    We convert raw seconds into a human-readable "Xh Ym" string.
    No external libraries needed; the kernel provides this for free.
    """
    try:
        with open("/proc/uptime", "r") as f:
            total_seconds = int(float(f.read().split()[0]))
        hours, remainder = divmod(total_seconds, 3600)
        minutes = remainder // 60
        return f"{hours}h {minutes:02d}m"
    except Exception:
        return "N/A"

# ─── Display ──────────────────────────────────────────────────────────────────

def load_font(size: int):
    """
    Loads a TrueType font at the given pixel size.
    DejaVuSans ships with Raspberry Pi OS — falls back to the built-in
    bitmap font if the file is missing so the script never crashes.
    """
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    ]
    for path in font_paths:
        try:
            return ImageFont.truetype(path, size)
        except IOError:
            continue
    return ImageFont.load_default()   # Last resort fallback


def draw_screen(device, ip: str, uptime: str) -> None:
    """
    Renders two lines of text to the OLED.

    Why only two lines: TrueType fonts include internal leading (spacing
    above/below glyphs). At size 13px the actual rendered height is ~16px,
    so two lines with an 18px gap sit cleanly without overlapping.

    Layout:
      Line 1 (y=2) : IP address
      Line 2 (y=34): System uptime
    """
    font = load_font(15)

    with Image.new("1", device.size, 0) as img:
        draw = ImageDraw.Draw(img)
        draw.text((0,  2), f"IP:{ip}",      font=font, fill=255)
        draw.text((0, 34), f"Up:{uptime}",  font=font, fill=255)
        device.display(img)

# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    # Set up the I2C serial link to the OLED, then initialise the SSD1306 driver
    serial = i2c(port=I2C_BUS, address=OLED_ADDRESS)
    device = ssd1306(serial)           # 128×64 by default

    print("Startup display running. Press Ctrl+C to stop.")

    while True:
        ip     = get_ip_address()
        uptime = get_uptime()
        draw_screen(device, ip, uptime)

        print(f"[OK] IP={ip}  Uptime={uptime}")   # Visible in journalctl logs
        time.sleep(UPDATE_INTERVAL)

if __name__ == "__main__":
    main()
