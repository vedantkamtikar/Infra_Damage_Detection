"""
airsim_takeoff_test.py

Tests basic AirSim connectivity and drone flight:
    - Connects to AirSim via Python API
    - Arms the drone and enables API control
    - Takes off to a target altitude
    - Hovers for a few seconds
    - Lands and disarms

Run this BEFORE any other drone script to verify the
Python <-> AirSim pipeline is working correctly.

Requirements:
    - Unreal Engine 4.27 with AirSim plugin must be running
    - DroneCV conda environment with airsim package installed

Usage:
    python scripts/airsim_takeoff_test.py
"""

import time
import airsim
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

TARGET_ALTITUDE = -3.0      # metres (AirSim uses NED: negative = up)
TAKEOFF_TIMEOUT = 15        # seconds to wait for takeoff to complete
HOVER_DURATION  = 5         # seconds to hover before landing
VEHICLE_NAME    = ""        # leave empty for default drone

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def print_state(client: airsim.MultirotorClient):
    """Print current drone position and state."""
    state = client.getMultirotorState()
    pos   = state.kinematics_estimated.position
    print(f"  Position  : x={pos.x_val:.2f}  y={pos.y_val:.2f}  z={pos.z_val:.2f}")
    print(f"  Armed     : {state.landed_state}")

# ─────────────────────────────────────────────
# MAIN FLIGHT SEQUENCE
# ─────────────────────────────────────────────

def run():
    print("\n" + "─" * 45)
    print("  AirSim Takeoff Test")
    print("─" * 45)

    # ── Step 1: Connect ──────────────────────
    print("\n[1/5] Connecting to AirSim...")
    try:
        client = airsim.MultirotorClient()
        client.confirmConnection()
        print("  ✓ Connected to AirSim")
    except Exception as e:
        print(f"  ✗ Connection failed: {e}")
        print("  Make sure Unreal Engine is running with AirSim plugin active.")
        return

    # ── Step 2: Enable API control ──────────
    print("\n[2/5] Enabling API control...")
    client.enableApiControl(True, VEHICLE_NAME)
    print("  ✓ API control enabled")

    # ── Step 3: Arm ──────────────────────────
    print("\n[3/5] Arming drone...")
    client.armDisarm(True, VEHICLE_NAME)
    print("  ✓ Drone armed")

    print("\n  State before takeoff:")
    print_state(client)

    # ── Step 4: Takeoff ──────────────────────
    print(f"\n[4/5] Taking off to {abs(TARGET_ALTITUDE)}m altitude...")
    try:
        client.takeoffAsync(timeout_sec=TAKEOFF_TIMEOUT, vehicle_name=VEHICLE_NAME).join()
        print("  ✓ Takeoff complete")
    except Exception as e:
        print(f"  ✗ Takeoff failed: {e}")
        _safe_land(client)
        return

    print("\n  State after takeoff:")
    print_state(client)

    # Move to target altitude
    print(f"\n  Climbing to {abs(TARGET_ALTITUDE)}m...")
    client.moveToZAsync(
        TARGET_ALTITUDE,
        velocity    = 1.5,
        timeout_sec = TAKEOFF_TIMEOUT,
        vehicle_name= VEHICLE_NAME
    ).join()
    print("  ✓ Target altitude reached")

    print("\n  State at target altitude:")
    print_state(client)

    # ── Step 5: Hover then land ──────────────
    print(f"\n[5/5] Hovering for {HOVER_DURATION} seconds...")
    client.hoverAsync(vehicle_name=VEHICLE_NAME)
    for i in range(HOVER_DURATION, 0, -1):
        print(f"  Hovering... {i}s remaining")
        time.sleep(1)

    print("\n  Landing...")
    _safe_land(client)

    print("\n" + "─" * 45)
    print("  ✓ Takeoff test complete — AirSim pipeline working!")
    print("─" * 45 + "\n")


def _safe_land(client: airsim.MultirotorClient):
    """Land and disarm safely."""
    try:
        client.landAsync(timeout_sec=15, vehicle_name=VEHICLE_NAME).join()
        print("  ✓ Landed")
    except Exception as e:
        print(f"  [!] Land error: {e}")
    try:
        client.armDisarm(False, VEHICLE_NAME)
        client.enableApiControl(False, VEHICLE_NAME)
        print("  ✓ Disarmed and API control released")
    except Exception as e:
        print(f"  [!] Disarm error: {e}")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    run()