import airsim
import cv2
import numpy as np
import sqlite3
from ultralytics import YOLO
from datetime import datetime

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
# MODEL
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

# ----------------------
# CAPTURE LOOP
# ----------------------
for i in range(10):  # take 10 frames

    responses = client.simGetImages([
        airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
    ])

    img1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
    img = img1d.reshape(responses[0].height, responses[0].width, 3)

    # ----------------------
    # DETECTION
    # ----------------------
    results = model(img, conf=0.15)

    for r in results:
        boxes = r.boxes

        for box in boxes:
            x, y, w, h = box.xywh[0].tolist()
            conf = float(box.conf[0])

            # SAVE TO DB
            cursor.execute("""
                INSERT INTO detections (x, y, w, h, confidence, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (x, y, w, h, conf, datetime.now().isoformat()))

    conn.commit()

    # SAVE IMAGE WITH BOXES
    plotted = r.plot()
    cv2.imwrite(f"outputs/frame_{i}.jpg", plotted)

print("Done")

client.landAsync().join()
client.armDisarm(False)
client.enableApiControl(False)
conn.close()