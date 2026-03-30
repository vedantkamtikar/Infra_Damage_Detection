import airsim
import math
import time
import sqlite3
import cv2
import numpy as np
from ultralytics import YOLO
import threading

# Global flag to let the background thread know when to stop
is_orbiting = True

def run_vision_and_logging():
    """Background thread for YOLO inference and SQL logging."""
    target_fps = 18 
    frame_delay = 1.0 / target_fps
    vision_client = airsim.MultirotorClient()
    vision_client.confirmConnection()
    
    model = YOLO("models/best_run8.pt")
    
    conn = sqlite3.connect("detections.db")
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS detections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp REAL,
        class TEXT,
        confidence REAL,
        x REAL,
        y REAL,
        z REAL,
        frame BLOB
    )
    """)
    conn.commit()

    # --- THROTTLE SETTINGS ---
    last_log_time = 0
    log_cooldown = 2.0  # Only log to DB once every 2 seconds

    while is_orbiting:
        start_vision_time = time.time()
        
        responses = vision_client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
        ])
        
        if responses and len(responses) > 0:
            response = responses[0]
            img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
            img_bgr = img1d.reshape(response.height, response.width, 3)

            # UPDATED: Changed conf from 0.25 to 0.3 to exclude low-confidence detections
            results = model(img_bgr, conf=0.3, verbose=False)
            
            # .copy() prevents the "readonly" OpenCV error
            annotated_frame = results[0].plot().copy() 

            state = vision_client.getMultirotorState()
            pos = state.kinematics_estimated.position
            
            detections_found = False
            max_conf = 0.0 
            current_time = time.time()

            for r in results:
                for box in r.boxes:
                    conf = float(box.conf[0])
                    label = model.names[int(box.cls[0])]

                    if conf > max_conf:
                        max_conf = conf
                    
                    detections_found = True

                    # Only write to DB if cooldown has passed
                    if current_time - last_log_time > log_cooldown:
                        success, encoded_image = cv2.imencode('.jpg', img_bgr)
                        frame_blob = encoded_image.tobytes() if success else None

                        cursor.execute("""
                            INSERT INTO detections (timestamp, class, confidence, x, y, z, frame)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        """, (current_time, label, conf, pos.x_val, pos.y_val, pos.z_val, frame_blob))
                        
                        last_log_time = current_time
                        conn.commit()
            
            if detections_found:
                cv2.putText(annotated_frame, "Damage Detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(annotated_frame, "Logging Enabled" if (current_time - last_log_time < 0.1) else "Logging Cooldown", 
                            (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # --- ALWAYS SHOWN: CONFIDENCE BAR ---
            bar_x, bar_y = 20, 110
            bar_w, bar_h = 200, 20
            cv2.putText(annotated_frame, f"Confidence: {max_conf:.2f}", (bar_x, bar_y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.rectangle(annotated_frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
            fill_w = int(bar_w * max_conf)
            if fill_w > 0:
                cv2.rectangle(annotated_frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), (255, 191, 0), -1)
            cv2.rectangle(annotated_frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 1)

            cv2.imshow("Live Inspection Feed", annotated_frame)
            cv2.waitKey(1) 

        elapsed = time.time() - start_vision_time
        if elapsed < frame_delay:
            time.sleep(frame_delay - elapsed)

    cv2.destroyAllWindows()
    conn.close()

def main():
    global is_orbiting
    vision_thread = threading.Thread(target=run_vision_and_logging)
    vision_thread.start()

    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)

    # Orbit Parameters
    start_global_x, start_global_y = 19.0, 0.0
    center_x, center_y = -start_global_x, -start_global_y
    radius = math.sqrt(center_x**2 + center_y**2)
    flight_z, velocity = -10.0, 3.0
    omega = velocity / radius 
    orbit_time = (2 * math.pi) / omega

    print("Taking off...")
    client.takeoffAsync().join()
    client.moveToZAsync(flight_z, 3.0).join()

    start_angle = math.atan2(0 - center_y, 0 - center_x)
    start_time = time.time()
    
    while time.time() - start_time < orbit_time:
        t = time.time() - start_time
        theta = start_angle + (omega * t)
        vx = -radius * omega * math.sin(theta)
        vy = radius * omega * math.cos(theta)
        yaw_deg = math.degrees(theta + math.pi)
        
        client.moveByVelocityZAsync(vx, vy, flight_z, duration=0.1, 
                                    drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom, 
                                    yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=yaw_deg))
        time.sleep(0.05)

    is_orbiting = False 
    client.landAsync().join()
    client.armDisarm(False)
    client.enableApiControl(False)
    vision_thread.join()

if __name__ == "__main__":
    main()