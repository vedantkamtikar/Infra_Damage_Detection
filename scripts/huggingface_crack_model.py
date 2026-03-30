from ultralytics import YOLO
from huggingface_hub import hf_hub_download

# Download model from Hugging Face
model_path = hf_hub_download(
    repo_id="cazzz307/yolov8-crack-detection",
    filename="best.pt",
    token="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"  # optional if already logged in
)

# Load and use model
model = YOLO(model_path)
results = model.predict("sample_images\\pic2.png", conf=0.25)
results[0].show()
