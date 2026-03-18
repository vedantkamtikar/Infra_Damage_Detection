from roboflow import Roboflow

rf = Roboflow(api_key="f5I1QiTjJWbEGnIuG8TG")
project = rf.workspace("enim-sppgm").project("cracks-detection-xtbn8")
version = project.version(9)
dataset = version.download("yolov11", location="datasets/ENIM")
