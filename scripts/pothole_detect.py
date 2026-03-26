from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import cv2 # Optional, for more advanced saving/showing

# 1. Download/Path (Already done, but kept for context)
model_path = hf_hub_download(repo_id="Harisanth/Pothole-Finetuned-YOLOv8", filename="best.pt")

# 2. Load the model
model = YOLO(model_path)

# 3. Run inference
# We add 'save=True' to automatically save the plotted image to 'runs/detect/predict'
results = model.predict(source='sample_images/pune_pothole.jpeg', conf=0.28, save=True)

# 4. Display the results
for r in results:
    # This will open a window showing the image with bounding boxes
    # Note: On some remote servers/SSH, this might require a GUI backend
    r.show() 
    
    # Alternatively, if you want to save it to a specific path manually:
    # r.save(filename='pothole_result.jpg')