import torch
from ultralytics.nn.tasks import DetectionModel
import airsim
import time
import sqlite3
import cv2
import numpy as np
from ultralytics import YOLO
import threading
import math

# 1. Fix for PyTorch 2.6 security layer
torch.serialization.add_safe_globals([DetectionModel])

# ======================================================
# EXACT UNREAL WORLD COORDINATES
# ======================================================
UE_SPAWN = (1900.0, 0.0, 122.0)
UE_START = (3810.0, -2050.0, 500.0)
UE_END   = (3810.0, 6110.0, 500.0)

# ======================================================
# COORDINATE CONVERSION (World to AirSim Relative)
# ======================================================
def to_airsim_ned(target, spawn):
    # AirSim uses Meters and Z-Down (Negative Z is Up)
    x = (target[0] - spawn[0]) / 100.0
    y = (target[1] - spawn[1]) / 100.0
    z = -(target[2] - spawn[2]) / 100.0
    return (x, y, z)

PT_A = to_airsim_ned(UE_START, UE_SPAWN)
PT_B = to_airsim_ned(UE_END, UE_SPAWN)

is_running = True

def vision_thread_logic():
    global is_running
    vision_client = airsim.MultirotorClient()
    vision_client.confirmConnection()
    
    # Load your fine-tuned model
    model = YOLO("pothole_hf.pt").to('cuda')
    
    # Database initialization
    conn = sqlite3.connect("road_inspection.db")
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS potholes")
    cursor.execute("""
        CREATE TABLE potholes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL,
            confidence REAL,
            drone_x REAL, drone_y REAL, drone_z REAL,
            frame BLOB
        )
    """)
    conn.commit()

    last_log_time = 0

    while is_running:
        # Request RGB Frame from Camera 0
        responses = vision_client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
        ])
        
        if responses and responses[0].width > 0:
            response = responses[0]
            img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
            img_bgr = img1d.reshape(response.height, response.width, 3)
            height, width = img_bgr.shape[:2]

            # YOLO Processing
            results = model(img_bgr, conf=0.45, verbose=False)
            
            # Visual Feedback
            annotated_frame = results[0].plot()
            if len(results[0].boxes) > 0:
                # Add "Pothole Detected" text
                cv2.putText(annotated_frame, "Pothole Detected", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                # Confidence score bar
                max_conf = max(float(box.conf[0]) for box in results[0].boxes)
                bar_x = 10
                bar_y = 70
                bar_max_width = 200
                bar_height = 15
                bar_width = int(max_conf * bar_max_width)
                cv2.rectangle(annotated_frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (0, 255, 0), -1)
                # Add confidence text
                cv2.putText(annotated_frame, f"Conf: {max_conf:.2f}", (10, bar_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Pothole Inspection - Live Feed", annotated_frame)
            cv2.imwrite("latest_frame.jpg", annotated_frame)  # share frame with web UI
            cv2.waitKey(1)

            # Log detections with current drone position
            if len(results[0].boxes) > 0:
                if time.time() - last_log_time > 2:
                    state = vision_client.getMultirotorState()
                    pos = state.kinematics_estimated.position
                    _, frame_jpg = cv2.imencode('.jpg', annotated_frame)
                    frame_blob = frame_jpg.tobytes()
                    for box in results[0].boxes:
                        cursor.execute("""
                            INSERT INTO potholes (timestamp, confidence, drone_x, drone_y, drone_z, frame)
                            VALUES (?, ?, ?, ?, ?, ?)
                        """, (time.time(), float(box.conf[0]), pos.x_val, pos.y_val, pos.z_val, frame_blob))
                    conn.commit()
                    last_log_time = time.time()

    cv2.destroyAllWindows()
    conn.close()

def main():
    global is_running
    
    # Start Vision System
    threading.Thread(target=vision_thread_logic, daemon=True).start()

    # Flight System Setup
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)

    print(f"Status: Taking off...")
    client.takeoffAsync().join()
    
    # -1.05 = Pitch (60 degrees down)
    # -1.57 = Yaw (Opposite direction along the Y-axis)
    # 0 = Roll
    # Pitch -1.05 (60 deg down), Roll 0, Yaw 0 (Look straight ahead of the drone)
    new_orientation = airsim.to_quaternion(-1.05, 0, 0)
    client.simSetCameraPose("0", airsim.Pose(airsim.Vector3r(0, 0, 0), new_orientation))

    print(f"Status: Flying to Start Point A {PT_A}")
    # Change -90 to 90 to face the direction of Point B (Positive Y)
    client.moveToPositionAsync(PT_A[0], PT_A[1], PT_A[2], velocity=5, 
                               yaw_mode=airsim.YawMode(False, 90)).join()
    
    time.sleep(2) 

    print(f"Status: Inspecting Road to Point B {PT_B}")
    # Stay facing 90 degrees while scanning the road to Point B
    client.moveToPositionAsync(PT_B[0], PT_B[1], PT_B[2], velocity=3, 
                               yaw_mode=airsim.YawMode(False, 90)).join()

    print("Status: Mission Complete. Landing.")
    is_running = False
    time.sleep(1) # Allow final DB commit
    
    client.landAsync().join()
    client.armDisarm(False)
    client.enableApiControl(False)

if __name__ == "__main__":
    main()