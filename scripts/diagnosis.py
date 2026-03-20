# import os
# from pathlib import Path

# def main():
#     ROOT_DIR = Path(__file__).resolve().parent.parent
#     lbl_dir = ROOT_DIR / "datasets" / "CRACKS" / "train" / "labels"

#     widths, heights = [], []
#     for f in os.listdir(lbl_dir):
#         with open(lbl_dir / f) as file:
#             for line in file:
#                 parts = line.strip().split()
#                 if len(parts) == 5:
#                     widths.append(float(parts[3]))
#                     heights.append(float(parts[4]))

#     print(f"Total boxes : {len(widths)}")
#     print(f"Mean width  : {sum(widths)/len(widths):.3f}")
#     print(f"Mean height : {sum(heights)/len(heights):.3f}")
#     print(f"w > 0.8     : {sum(1 for w in widths if w > 0.8)}")
#     print(f"h > 0.8     : {sum(1 for h in heights if h > 0.8)}")
#     print(f"w < 0.3     : {sum(1 for w in widths if w < 0.3)}")
#     print(f"h < 0.3     : {sum(1 for h in heights if h < 0.3)}")

# if __name__ == "__main__":
#     main()

import os
import cv2
from ultralytics import YOLO

# ----------------------
# LOAD MODEL
# ----------------------
model = YOLO("models/best_run7.pt")

# ----------------------
# PATHS
# ----------------------
import airsim
import cv2
import numpy as np

# Connect to AirSim
client = airsim.MultirotorClient()
client.confirmConnection()

# Takeoff (optional but good)
client.enableApiControl(True)
client.armDisarm(True)
client.takeoffAsync().join()

while True:

    # Get image from drone camera
    response = client.simGetImages([
        airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
    ])[0]

    # Safety check
    if response.height == 0 or response.width == 0:
        print("Empty frame")
        continue

    # Convert to OpenCV format
    img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
    img = img1d.reshape(response.height, response.width, 3)

    # Show image
    cv2.imshow("Drone Camera", img)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

# Cleanup
client.landAsync().join()
client.armDisarm(False)
client.enableApiControl(False)