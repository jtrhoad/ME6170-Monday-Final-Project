#!/usr/bin/env python3
"""
battery_diagnostic.py
---------------------
Run this to find the correct raw values from the battery board at 0x2B.
Check your battery voltage with a multimeter first, then compare to the
raw readings here to calculate the right conversion factor.

Usage:
    python3.11 battery_diagnostic.py
"""

import smbus2

ADDRESS = 0x2B
BUS     = 1

bus = smbus2.SMBus(BUS)

print("=" * 50)
print(f"Scanning registers on device 0x{ADDRESS:02X}")
print("=" * 50)

# Try reading single bytes from the first 8 registers
print("\n-- Single byte reads (read_byte_data) --")
for reg in range(0x00, 0x08):
    try:
        val = bus.read_byte_data(ADDRESS, reg)
        print(f"  Register 0x{reg:02X}: raw={val:3d}  (0x{val:02X})  as voltage guess: {val * 0.1:.2f}V")
    except OSError:
        print(f"  Register 0x{reg:02X}: no response")

# Try reading 2-byte words from the first 4 registers
print("\n-- Word reads (read_word_data) --")
for reg in range(0x00, 0x04):
    try:
        raw = bus.read_word_data(ADDRESS, reg)
        swapped = ((raw & 0xFF) << 8) | ((raw >> 8) & 0xFF)
        print(f"  Register 0x{reg:02X}: raw={raw:6d}  swapped={swapped:6d}"
              f"  current formula: {raw * 0.00489:.2f}V  swapped: {swapped * 0.00489:.2f}V")
    except OSError:
        print(f"  Register 0x{reg:02X}: no response")

# Try reading a 2-byte block
print("\n-- Block reads (read_i2c_block_data, 2 bytes) --")
for reg in range(0x00, 0x04):
    try:
        data = bus.read_i2c_block_data(ADDRESS, reg, 2)
        combined_be = (data[0] << 8) | data[1]   # big-endian
        combined_le = (data[1] << 8) | data[0]   # little-endian
        print(f"  Register 0x{reg:02X}: bytes={data}  "
              f"big-endian={combined_be}  little-endian={combined_le}")
    except OSError:
        print(f"  Register 0x{reg:02X}: no response")

bus.close()

print("\n" + "=" * 50)
print("Next step:")
print("  1. Measure your battery voltage with a multimeter")
print("  2. Find which register/value looks most like a battery reading")
print("  3. new_factor = actual_voltage / raw_value")
print("=" * 50)
