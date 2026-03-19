from roboflow import Roboflow
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

rf = Roboflow(api_key="Lkl8pHiBySPUXrW5xGli")
project = rf.workspace("roberta-ruggiero-8apkt").project("cracks-wnd2x")
version = project.version(2)
dataset = version.download("yolov8", location=str(ROOT_DIR / "datasets" / "CRACKS"))