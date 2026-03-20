import airsim
import cv2
import numpy as np
import sqlite3
from ultralytics import YOLO
from datetime import datetime
import os
import time

# ----------------------
# SETUP
# ----------------------
os.makedirs("outputs", exist_ok=True)

# ----------------------
# DB SETUP
# ----------------------
conn = sqlite3.connect("detections.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    x REAL,
    y REAL,
    w REAL,
    h REAL,
    confidence REAL,
    timestamp TEXT
)
""")

# ----------------------
# LOGGING CONTROL
# ----------------------
last_logged_time = time.time()
LOG_INTERVAL = 3
detection_count = 0

# ----------------------
# LOAD MODEL
# ----------------------
model = YOLO("models/best_run7.pt")

# ----------------------
# AIRSIM SETUP
# ----------------------
client = airsim.MultirotorClient()
client.confirmConnection()

client.enableApiControl(True)
client.armDisarm(True)

client.takeoffAsync().join()
client.moveToZAsync(-5, 1).join()

# ----------------------
# MOVEMENT STATE
# ----------------------
moving = True

# ----------------------
# LOOP
# ----------------------
for i in range(1000):

    # ----------------------
    # MOVE LEFT UNTIL DETECTED
    # ----------------------
    if moving:
        client.moveByVelocityAsync(0, -1, 0, 0.3)

    # ----------------------
    # GET IMAGE
    # ----------------------
    response = client.simGetImages([
        airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
    ])[0]

    if response.height == 0 or response.width == 0:
        continue

    img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
    img = img1d.reshape(response.height, response.width, 3)

    img = cv2.resize(img, (1280, 720))
    display_img = img.copy()

    # ----------------------
    # YOLO DETECTION
    # ----------------------
    results = model(img, conf=0.15)  # slightly higher for stability
    boxes = results[0].boxes

    if len(boxes) > 0:

        best_box = max(boxes, key=lambda b: float(b.conf[0]))

        x, y, w, h = best_box.xywh[0].tolist()
        conf = float(best_box.conf[0])

        # Stop movement when detected
        moving = False
        client.moveByVelocityAsync(0, 0, 0, 0.1)

        # Tight box
        w *= 0.5
        h *= 0.8

        x1 = int(x - w / 2)
        y1 = int(y - h / 2)
        x2 = int(x + w / 2)
        y2 = int(y + h / 2)

        # Draw box
        cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # ----------------------
        # LOGGING
        # ----------------------
        current_time = time.time()

        if current_time - last_logged_time > LOG_INTERVAL:
            cursor.execute("""
                INSERT INTO detections (x, y, w, h, confidence, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (x, y, w, h, conf, datetime.now().isoformat()))

            conn.commit()
            last_logged_time = current_time
            detection_count += 1

            print(f"Logged: conf={conf:.2f}")

        # Overlay
        cv2.putText(display_img, "CRACK DETECTED - STOPPED",
                    (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)

        cv2.putText(display_img, f"Confidence: {conf:.2f}",
                    (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 0), 2)

    else:
        cv2.putText(display_img, "SCANNING...",
                    (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 200, 200), 2)

    # Always show count
    cv2.putText(display_img, f"Detections Logged: {detection_count}",
                (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 255), 2)

    # ----------------------
    # DISPLAY
    # ----------------------
    cv2.imshow("Drone Crack Detection", display_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ----------------------
# CLEANUP
# ----------------------
cv2.destroyAllWindows()

client.landAsync().join()
client.armDisarm(False)
client.enableApiControl(False)

conn.close()

print("✅ Done!")