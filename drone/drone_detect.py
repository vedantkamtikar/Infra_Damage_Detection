"""
drone_detect.py

Autonomous drone inspection pipeline:
    AirSim flight (circular orbit) → RGB frame capture → YOLO11m inference → SQLite DB logging

The drone flies a circular orbit around a target building, hovering at each waypoint.
During each hover, frames are captured and passed through the trained YOLO model.
Detections above CONF_THRESHOLD are logged to the SQLite database.
Annotated frames are optionally saved to results/frames/.

Requirements:
    - Unreal Engine 4.27 with AirSim plugin must be running
    - DroneCV conda environment with airsim package installed
    - Trained weights at models/best_run6.pt (or update WEIGHTS below)
    - db.py must be in the drone/ folder

Usage:
    python drone/drone_detect.py
"""

import time
import math
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path

import airsim
from ultralytics import YOLO

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from drone.db import DetectionDB

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

ROOT_DIR        = Path(__file__).resolve().parent.parent
WEIGHTS         = ROOT_DIR / "models" / "best_run7.pt"

# Detection
CONF_THRESHOLD  = 0.35        # minimum confidence to log a detection
SAVE_FRAMES     = True        # save annotated frames to results/frames/
FRAME_DIR       = ROOT_DIR / "results" / "frames"
FRAMES_PER_HOVER= 3           # number of frames to capture at each waypoint hover

# Orbit parameters
ORBIT_CENTER_X  = 0.0         # x coordinate of orbit center (metres)
ORBIT_CENTER_Y  = 0.0         # y coordinate of orbit center (metres)
ORBIT_RADIUS    = 10.0        # radius of the circular orbit (metres)
ORBIT_ALTITUDE  = -5.0        # flight altitude in NED (negative = up)
NUM_WAYPOINTS   = 8           # number of waypoints around the circle
HOVER_DURATION  = 3           # seconds to hover at each waypoint

# Flight parameters
FLIGHT_SPEED    = 3.0         # m/s between waypoints
TAKEOFF_TIMEOUT = 15          # seconds
WAYPOINT_TIMEOUT= 30          # seconds to reach each waypoint

CAMERA_NAME     = "front_center"
VEHICLE_NAME    = ""          # leave empty for default drone

# ─────────────────────────────────────────────
# FLIGHT HELPERS
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


def get_position(client: airsim.MultirotorClient) -> tuple:
    """Returns current NED position as (x, y, z)."""
    state = client.getMultirotorState()
    pos = state.kinematics_estimated.position
    return (pos.x_val, pos.y_val, pos.z_val)


def print_position(client: airsim.MultirotorClient, label: str = ""):
    """Print current drone position."""
    x, y, z = get_position(client)
    tag = f"  [{label}]" if label else "  "
    print(f"{tag} x={x:.2f}  y={y:.2f}  z={z:.2f}")


def safe_land(client: airsim.MultirotorClient):
    """Land and disarm safely."""
    print("\n[→] Landing...")
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
# DETECTION HELPERS
# ─────────────────────────────────────────────

def get_frame(client: airsim.MultirotorClient) -> np.ndarray | None:
    """Capture a single RGB frame from the front camera."""
    try:
        responses = client.simGetImages([
            airsim.ImageRequest(CAMERA_NAME, airsim.ImageType.Scene, False, False)
        ])
        if not responses or responses[0].width == 0:
            return None
        img_1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
        img = img_1d.reshape(responses[0].height, responses[0].width, 3)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"  [!] Frame capture failed: {e}")
        return None


def run_inference(model: YOLO, frame: np.ndarray) -> list:
    """
    Runs YOLO inference on a frame.
    Returns list of detection dicts above CONF_THRESHOLD.
    """
    results = model.predict(frame, conf=CONF_THRESHOLD, verbose=False)
    detections = []
    for r in results:
        for box in r.boxes:
            conf     = float(box.conf[0])
            cls_id   = int(box.cls[0])
            cls_name = model.names[cls_id]
            xywhn    = box.xywhn[0].tolist()
            detections.append({
                "class_id"   : cls_id,
                "class_name" : cls_name,
                "confidence" : conf,
                "bbox_x"     : xywhn[0],
                "bbox_y"     : xywhn[1],
                "bbox_w"     : xywhn[2],
                "bbox_h"     : xywhn[3],
            })
    return detections


def annotate_frame(frame: np.ndarray, detections: list,
                   frame_id: int, position: tuple) -> np.ndarray:
    """Draws bounding boxes and HUD info onto the frame."""
    h, w = frame.shape[:2]
    annotated = frame.copy()

    for det in detections:
        cx = int(det["bbox_x"] * w)
        cy = int(det["bbox_y"] * h)
        bw = int(det["bbox_w"] * w)
        bh = int(det["bbox_h"] * h)
        x1, y1 = cx - bw // 2, cy - bh // 2
        x2, y2 = cx + bw // 2, cy + bh // 2

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
        label = f"{det['class_name']} {det['confidence']:.2f}"
        cv2.putText(annotated, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # HUD overlay
    ned_x, ned_y, ned_z = position
    hud = [
        f"Frame : {frame_id:04d}",
        f"NED   : ({ned_x:.1f}, {ned_y:.1f}, {ned_z:.1f})",
        f"Dets  : {len(detections)}",
        datetime.now().strftime("%H:%M:%S"),
    ]
    for i, line in enumerate(hud):
        cv2.putText(annotated, line, (10, 20 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return annotated


def capture_and_detect(client, model, db, frame_id, wp_num):
    """
    Captures FRAMES_PER_HOVER frames during a hover window,
    runs inference on each, logs detections to DB.
    Returns updated frame_id and total detections at this waypoint.
    """
    wp_detections = 0
    interval = HOVER_DURATION / FRAMES_PER_HOVER

    for f in range(FRAMES_PER_HOVER):
        frame = get_frame(client)
        if frame is None:
            print(f"    [!] Frame {frame_id} capture failed — skipping")
            frame_id += 1
            time.sleep(interval)
            continue

        pos        = get_position(client)
        detections = run_inference(model, frame)

        if detections:
            annotated  = annotate_frame(frame, detections, frame_id, pos)
            img_path   = str(FRAME_DIR / f"frame_{frame_id:04d}.jpg")
            cv2.imwrite(img_path, annotated)
            db.log(frame_id, pos, detections, img_path)
            wp_detections += len(detections)
            print(f"    Frame {frame_id:04d} — {len(detections)} detection(s) "
                  f"[conf: {max(d['confidence'] for d in detections):.2f}]")
        else:
            print(f"    Frame {frame_id:04d} — no detections")

        frame_id += 1
        time.sleep(interval)

    return frame_id, wp_detections


# ─────────────────────────────────────────────
# MAIN MISSION
# ─────────────────────────────────────────────

def run():
    print("\n" + "─" * 50)
    print("  Drone Inspection Pipeline")
    print("─" * 50)
    print(f"  Weights        : {WEIGHTS}")
    print(f"  Conf threshold : {CONF_THRESHOLD}")
    print(f"  Orbit radius   : {ORBIT_RADIUS}m")
    print(f"  Altitude       : {abs(ORBIT_ALTITUDE)}m")
    print(f"  Waypoints      : {NUM_WAYPOINTS}")
    print(f"  Frames/hover   : {FRAMES_PER_HOVER}")
    print(f"  Hover duration : {HOVER_DURATION}s")
    print(f"  Save frames    : {SAVE_FRAMES}")
    print("─" * 50)

    if not WEIGHTS.exists():
        raise FileNotFoundError(
            f"Weights not found: {WEIGHTS}\n"
            f"Train the model first or update WEIGHTS path."
        )

    if SAVE_FRAMES:
        FRAME_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load model ───────────────────────────
    print("\n[1/6] Loading YOLO model...")
    model = YOLO(str(WEIGHTS))
    print("  ✓ Model loaded")

    # ── Connect to DB ────────────────────────
    print("\n[2/6] Connecting to database...")
    db = DetectionDB()
    db.clear()
    print("  ✓ Database ready")

    # ── Connect to AirSim ────────────────────
    print("\n[3/6] Connecting to AirSim...")
    try:
        client = airsim.MultirotorClient()
        client.confirmConnection()
        print("  ✓ Connected")
    except Exception as e:
        print(f"  ✗ Connection failed: {e}")
        print("  Make sure Unreal Engine is running with AirSim active.")
        db.close()
        return

    # ── Arm and takeoff ──────────────────────
    print("\n[4/6] Arming and taking off...")
    client.enableApiControl(True, VEHICLE_NAME)
    client.armDisarm(True, VEHICLE_NAME)

    try:
        client.takeoffAsync(
            timeout_sec  = TAKEOFF_TIMEOUT,
            vehicle_name = VEHICLE_NAME
        ).join()
    except Exception as e:
        print(f"  ✗ Takeoff failed: {e}")
        safe_land(client)
        db.close()
        return

    print(f"  Climbing to {abs(ORBIT_ALTITUDE)}m...")
    client.moveToZAsync(
        ORBIT_ALTITUDE,
        velocity     = 2.0,
        timeout_sec  = TAKEOFF_TIMEOUT,
        vehicle_name = VEHICLE_NAME
    ).join()
    print("  ✓ At orbit altitude")
    print_position(client, "takeoff")

    # ── Fly to orbit start ───────────────────
    waypoints = generate_orbit_waypoints(
        ORBIT_CENTER_X, ORBIT_CENTER_Y,
        ORBIT_RADIUS, ORBIT_ALTITUDE,
        NUM_WAYPOINTS
    )

    print(f"\n[5/6] Flying to orbit start...")
    sx, sy, sz = waypoints[0]
    client.moveToPositionAsync(
        sx, sy, sz,
        velocity     = FLIGHT_SPEED,
        timeout_sec  = WAYPOINT_TIMEOUT,
        vehicle_name = VEHICLE_NAME
    ).join()
    print("  ✓ At orbit start")
    print_position(client, "orbit start")

    # ── Execute orbit with detection ─────────
    print(f"\n[6/6] Executing orbit — {NUM_WAYPOINTS} waypoints...\n")

    frame_id         = 0
    total_detections = 0

    try:
        for i, (wx, wy, wz) in enumerate(waypoints):
            wp_num = i + 1
            print(f"  Waypoint {wp_num}/{NUM_WAYPOINTS} → "
                  f"({wx:.1f}, {wy:.1f}, {abs(wz):.1f}m)")

            # Fly to waypoint
            client.moveToPositionAsync(
                wx, wy, wz,
                velocity     = FLIGHT_SPEED,
                timeout_sec  = WAYPOINT_TIMEOUT,
                vehicle_name = VEHICLE_NAME
            ).join()

            print_position(client, f"wp{wp_num}")

            # Hover and capture frames for detection
            print(f"    Hovering {HOVER_DURATION}s — "
                  f"capturing {FRAMES_PER_HOVER} frames...")
            client.hoverAsync(vehicle_name=VEHICLE_NAME)

            frame_id, wp_dets = capture_and_detect(
                client, model, db, frame_id, wp_num
            )
            total_detections += wp_dets
            print(f"    Waypoint {wp_num} done — {wp_dets} detection(s)\n")

        # Complete the loop
        print("  Returning to orbit start to complete loop...")
        client.moveToPositionAsync(
            waypoints[0][0], waypoints[0][1], waypoints[0][2],
            velocity     = FLIGHT_SPEED,
            timeout_sec  = WAYPOINT_TIMEOUT,
            vehicle_name = VEHICLE_NAME
        ).join()
        print("  ✓ Orbit complete")

    except KeyboardInterrupt:
        print("\n[!] Flight interrupted by user")

    finally:
        safe_land(client)
        db.summary()
        db.close()

        print("\n" + "─" * 50)
        print("  Mission Complete")
        print("─" * 50)
        print(f"  Total frames     : {frame_id}")
        print(f"  Total detections : {total_detections}")
        print(f"  Database         : results/detections.db")
        if SAVE_FRAMES:
            print(f"  Frames saved     : {FRAME_DIR}")
        print("─" * 50 + "\n")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    run()