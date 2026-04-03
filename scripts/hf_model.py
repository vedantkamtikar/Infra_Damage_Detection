from huggingface_hub import hf_hub_download

# This downloads the 'best.pt' file directly into your current folder
model_path = hf_hub_download(
    repo_id="cazzz307/Pothole-Finetuned-YoloV8", 
    filename="best.pt",
    local_dir=".",        # Downloads to current directory
    local_dir_use_symlinks=False, # Ensures it's a real file, not a link
    token="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"
)

print(f"Model downloaded to: {model_path}")