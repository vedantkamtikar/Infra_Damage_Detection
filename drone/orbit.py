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
    """This runs completely in the background, out of the way of your flight loop."""
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
        z REAL
    )
    """)
    conn.commit()

    while is_orbiting:
        start_vision_time = time.time()
        
        # Using Camera "0" as requested
        responses = vision_client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
        ])
        
        if responses and len(responses) > 0:
            response = responses[0]
            img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
            
            # AirSim natively outputs BGR in this array, so we just reshape it and hand it to YOLO
            img_bgr = img1d.reshape(response.height, response.width, 3)

            results = model(img_bgr, conf=0.25, verbose=False)

            annotated_frame = results[0].plot() 
            cv2.imshow("Structural Damage Detection - Live View", annotated_frame)
            cv2.waitKey(1) 

            state = vision_client.getMultirotorState()
            pos = state.kinematics_estimated.position
            
            detections_found = False

            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = model.names[cls]

                    cursor.execute("""
                        INSERT INTO detections (timestamp, class, confidence, x, y, z)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (time.time(), label, conf, pos.x_val, pos.y_val, pos.z_val))
                    
                    detections_found = True
            
            if detections_found:
                conn.commit()

        # FIX: Pushed the throttle to 30 FPS (0.033 seconds per frame)
        elapsed = time.time() - start_vision_time
        if elapsed < frame_delay:
            time.sleep(frame_delay - elapsed)

    # Cleanup when is_orbiting becomes False
    cv2.destroyAllWindows()
    conn.close()

def main():
    global is_orbiting
    
    # Start the YOLO vision script in a background thread
    vision_thread = threading.Thread(target=run_vision_and_logging)
    vision_thread.start()

    # ==============================
    # YOUR EXACT UNCHANGED LOOP
    # ==============================
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)

    start_global_x = 19.00
    start_global_y = 0.00
    center_x = 0.00 - start_global_x
    center_y = 0.00 - start_global_y

    radius = math.sqrt(center_x**2 + center_y**2)
    flight_z = -10.0
    velocity = 3.0 # meters per second
    
    omega = velocity / radius 
    orbit_time = (2 * math.pi) / omega

    print("Taking off...")
    client.takeoffAsync().join()
    client.moveToZAsync(flight_z, 3.0).join()

    start_angle = math.atan2(0 - center_y, 0 - center_x)

    print(f"Starting POI orbit. Estimated time: {orbit_time:.1f} seconds...")
    start_time = time.time()
    
    # Control loop running at roughly 20Hz
    while time.time() - start_time < orbit_time:
        t = time.time() - start_time
        
        theta = start_angle + (omega * t)
        
        vx = -radius * omega * math.sin(theta)
        vy = radius * omega * math.cos(theta)
        
        yaw_deg = math.degrees(theta + math.pi)
        
        client.moveByVelocityZAsync(
            vx, vy, flight_z, 
            duration=0.1, 
            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom, 
            yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=yaw_deg)
        )
        
        time.sleep(0.05)

    print("Orbit complete. Stopping...")
    
    # Signal the background thread to shut down safely
    is_orbiting = False 

    client.moveByVelocityAsync(0, 0, 0, 1).join()
    client.moveToPositionAsync(0, 0, flight_z, velocity, yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=0)).join()
    client.landAsync().join()
    client.armDisarm(False)
    client.enableApiControl(False)
    
    # Wait for the vision thread to finish its last image before exiting completely
    vision_thread.join()
    print("Done.")

if __name__ == "__main__":
    main()