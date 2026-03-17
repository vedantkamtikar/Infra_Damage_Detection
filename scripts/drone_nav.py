"""
drone_navigate.py

Autonomous waypoint-based drone navigation using AirSim.
Flies the drone in a circular orbit around a target point,
hovering briefly at each waypoint before continuing.

Requirements:
    - Unreal Engine 4.27 with AirSim plugin must be running
    - DroneCV conda environment with airsim package installed
    - Run airsim_takeoff_test.py first to verify connection

Usage:
    python scripts/drone_navigate.py
"""

import time
import math
import airsim
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

# Orbit parameters
ORBIT_CENTER_X  = 0.0       # x coordinate of orbit center (metres)
ORBIT_CENTER_Y  = 0.0       # y coordinate of orbit center (metres)
ORBIT_RADIUS    = 10.0      # radius of the circular orbit (metres)
ORBIT_ALTITUDE  = -5.0      # flight altitude in NED (negative = up)
NUM_WAYPOINTS   = 8         # number of waypoints around the circle
HOVER_DURATION  = 2         # seconds to hover at each waypoint

# Flight parameters
FLIGHT_SPEED    = 3.0       # m/s between waypoints
TAKEOFF_ALT     = -3.0      # initial takeoff altitude (NED)
TAKEOFF_TIMEOUT = 15        # seconds
WAYPOINT_TIMEOUT= 30        # seconds to reach each waypoint

VEHICLE_NAME    = ""        # leave empty for default drone

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def generate_orbit_waypoints(cx, cy, radius, altitude, n_points):
    """Generate evenly spaced waypoints in a circle around (cx, cy)."""
    waypoints = []
    for i in range(n_points):
        angle = 2 * math.pi * i / n_points
        x = cx + radius * math.cos(angle)
        y = cy + radius * math.sin(angle)
        waypoints.append((x, y, altitude))
    return waypoints


def print_position(client: airsim.MultirotorClient, label: str = ""):
    """Print current drone position."""
    state = client.getMultirotorState()
    pos   = state.kinematics_estimated.position
    tag   = f"  [{label}]" if label else " "
    print(f"{tag} x={pos.x_val:.2f}  y={pos.y_val:.2f}  z={pos.z_val:.2f}")


def safe_land(client: airsim.MultirotorClient):
    """Land and disarm safely."""
    print("\n  Landing...")
    try:
        client.landAsync(timeout_sec=15, vehicle_name=VEHICLE_NAME).join()
        print("  ✓ Landed")
    except Exception as e:
        print(f"  [!] Land error: {e}")
    try:
        client.armDisarm(False, VEHICLE_NAME)
        client.enableApiControl(False, VEHICLE_NAME)
        print("  ✓ Disarmed")
    except Exception as e:
        print(f"  [!] Disarm error: {e}")


# ─────────────────────────────────────────────
# MAIN MISSION
# ─────────────────────────────────────────────

def run():
    print("\n" + "─" * 45)
    print("  Drone Circular Orbit Navigation")
    print("─" * 45)
    print(f"  Orbit center  : ({ORBIT_CENTER_X}, {ORBIT_CENTER_Y})")
    print(f"  Orbit radius  : {ORBIT_RADIUS}m")
    print(f"  Altitude      : {abs(ORBIT_ALTITUDE)}m")
    print(f"  Waypoints     : {NUM_WAYPOINTS}")
    print(f"  Hover time    : {HOVER_DURATION}s per waypoint")
    print(f"  Flight speed  : {FLIGHT_SPEED} m/s")
    print("─" * 45)

    # ── Step 1: Connect ──────────────────────
    print("\n[1/5] Connecting to AirSim...")
    try:
        client = airsim.MultirotorClient()
        client.confirmConnection()
        print("  ✓ Connected")
    except Exception as e:
        print(f"  ✗ Connection failed: {e}")
        print("  Make sure Unreal Engine is running with AirSim active.")
        return

    # ── Step 2: Arm and enable API ───────────
    print("\n[2/5] Arming drone...")
    client.enableApiControl(True, VEHICLE_NAME)
    client.armDisarm(True, VEHICLE_NAME)
    print("  ✓ Armed")

    # ── Step 3: Takeoff ──────────────────────
    print(f"\n[3/5] Taking off...")
    try:
        client.takeoffAsync(timeout_sec=TAKEOFF_TIMEOUT, vehicle_name=VEHICLE_NAME).join()
    except Exception as e:
        print(f"  ✗ Takeoff failed: {e}")
        safe_land(client)
        return

    # Climb to orbit altitude
    print(f"  Climbing to {abs(ORBIT_ALTITUDE)}m...")
    client.moveToZAsync(
        ORBIT_ALTITUDE,
        velocity     = 2.0,
        timeout_sec  = TAKEOFF_TIMEOUT,
        vehicle_name = VEHICLE_NAME
    ).join()
    print("  ✓ At orbit altitude")
    print_position(client, "takeoff")

    # ── Step 4: Fly to orbit start ───────────
    waypoints = generate_orbit_waypoints(
        ORBIT_CENTER_X, ORBIT_CENTER_Y,
        ORBIT_RADIUS, ORBIT_ALTITUDE,
        NUM_WAYPOINTS
    )

    print(f"\n[4/5] Flying to orbit start point...")
    start_x, start_y, start_z = waypoints[0]
    client.moveToPositionAsync(
        start_x, start_y, start_z,
        velocity     = FLIGHT_SPEED,
        timeout_sec  = WAYPOINT_TIMEOUT,
        vehicle_name = VEHICLE_NAME
    ).join()
    print(f"  ✓ At orbit start")
    print_position(client, "orbit start")

    # ── Step 5: Execute orbit ─────────────────
    print(f"\n[5/5] Executing circular orbit ({NUM_WAYPOINTS} waypoints)...\n")

    for i, (wx, wy, wz) in enumerate(waypoints):
        wp_num = i + 1
        print(f"  Waypoint {wp_num}/{NUM_WAYPOINTS} → ({wx:.1f}, {wy:.1f}, {abs(wz):.1f}m)")

        # Fly to waypoint
        client.moveToPositionAsync(
            wx, wy, wz,
            velocity     = FLIGHT_SPEED,
            timeout_sec  = WAYPOINT_TIMEOUT,
            vehicle_name = VEHICLE_NAME
        ).join()

        print_position(client, f"wp{wp_num}")

        # Hover briefly
        print(f"    Hovering for {HOVER_DURATION}s...")
        client.hoverAsync(vehicle_name=VEHICLE_NAME)
        time.sleep(HOVER_DURATION)

    # Return to start and complete orbit
    print(f"\n  Returning to orbit start to complete loop...")
    client.moveToPositionAsync(
        waypoints[0][0], waypoints[0][1], waypoints[0][2],
        velocity     = FLIGHT_SPEED,
        timeout_sec  = WAYPOINT_TIMEOUT,
        vehicle_name = VEHICLE_NAME
    ).join()
    print("  ✓ Orbit complete")

    # ── Land ─────────────────────────────────
    safe_land(client)

    print("\n" + "─" * 45)
    print("  ✓ Mission complete")
    print("─" * 45 + "\n")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    run()