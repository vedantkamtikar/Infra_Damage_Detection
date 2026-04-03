"""
server.py

FastAPI backend for the Infra Damage Detection web UI.

Endpoints:
    GET  /logs             — detection logs (paginated, with base64 frame images)
    GET  /stats            — aggregate detection statistics
    GET  /training/metrics — training results.csv as JSON
    GET  /training/images  — list evaluation plot images
    POST /detect           — upload an image for on-demand YOLO inference
    GET  /mission          — mission/orbit configuration info

DB schema (from orbit.py):
    detections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp REAL,
        class TEXT,
        confidence REAL,
        x REAL, y REAL, z REAL,
        frame BLOB
    )

Usage:
    uvicorn scripts.server:app --reload --port 8000
    (or)  python -m uvicorn scripts.server:app --reload --port 8000
"""

from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import sqlite3
import base64
import csv
import io
import os
import time
from pathlib import Path

app = FastAPI(title="Infraspect Detection API")

# Allow React dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ROOT_DIR = Path(__file__).resolve().parent.parent
DB_PATH  = ROOT_DIR / "detections.db"
ROAD_DB_PATH = ROOT_DIR / "road_inspection.db"


def get_db():
    """Get a connection to the detections database."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def get_road_db():
    """Get a connection to the road inspection database."""
    conn = sqlite3.connect(str(ROAD_DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


# ─────────────────────────────────────────────
# DETECTION LOGS
# ─────────────────────────────────────────────

@app.get("/logs")
def get_logs(
    limit: int = Query(20, ge=1, le=200),
    offset: int = Query(0, ge=0),
    min_conf: float = Query(0.0, ge=0.0, le=1.0),
):
    """Return detection logs with base64-encoded frame images."""
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute(
        """SELECT id, timestamp, class, confidence, x, y, z, frame
           FROM detections
           WHERE confidence >= ?
           ORDER BY id DESC
           LIMIT ? OFFSET ?""",
        (min_conf, limit, offset),
    )
    rows = cursor.fetchall()

    # Total count for pagination
    cursor.execute(
        "SELECT COUNT(*) FROM detections WHERE confidence >= ?",
        (min_conf,),
    )
    total = cursor.fetchone()[0]
    conn.close()

    output = []
    for row in rows:
        image_b64 = ""
        if row["frame"]:
            image_b64 = base64.b64encode(row["frame"]).decode("utf-8")

        output.append({
            "id": row["id"],
            "timestamp": row["timestamp"],
            "label": row["class"],
            "confidence": round(row["confidence"], 4),
            "location": {
                "x": round(row["x"], 2),
                "y": round(row["y"], 2),
                "z": round(row["z"], 2),
            },
            "image": f"data:image/jpeg;base64,{image_b64}" if image_b64 else None,
        })

    return {"logs": output, "total": total, "limit": limit, "offset": offset}


# ─────────────────────────────────────────────
# AGGREGATE STATS
# ─────────────────────────────────────────────

@app.get("/stats")
def get_stats():
    """Aggregate detection statistics."""
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM detections")
    total = cursor.fetchone()[0]

    if total == 0:
        conn.close()
        return {
            "total_detections": 0,
            "avg_confidence": 0,
            "max_confidence": 0,
            "min_confidence": 0,
            "unique_classes": [],
            "class_counts": {},
            "confidence_distribution": [],
        }

    cursor.execute("SELECT AVG(confidence), MAX(confidence), MIN(confidence) FROM detections")
    avg_c, max_c, min_c = cursor.fetchone()

    cursor.execute("SELECT DISTINCT class FROM detections")
    classes = [r[0] for r in cursor.fetchall()]

    cursor.execute("SELECT class, COUNT(*) FROM detections GROUP BY class")
    class_counts = {r[0]: r[1] for r in cursor.fetchall()}

    # Confidence histogram (10 bins)
    bins = []
    for i in range(10):
        lo = i * 0.1
        hi = (i + 1) * 0.1
        cursor.execute(
            "SELECT COUNT(*) FROM detections WHERE confidence >= ? AND confidence < ?",
            (lo, hi if i < 9 else 1.01),
        )
        bins.append({"range": f"{lo:.1f}-{hi:.1f}", "count": cursor.fetchone()[0]})

    conn.close()
    return {
        "total_detections": total,
        "avg_confidence": round(avg_c, 4),
        "max_confidence": round(max_c, 4),
        "min_confidence": round(min_c, 4),
        "unique_classes": classes,
        "class_counts": class_counts,
        "confidence_distribution": bins,
    }


# ─────────────────────────────────────────────
# POTHOLE DETECTION LOGS (road_inspection.db)
# ─────────────────────────────────────────────

@app.get("/pothole-logs")
def get_pothole_logs(
    limit: int = Query(20, ge=1, le=200),
    offset: int = Query(0, ge=0),
    min_conf: float = Query(0.0, ge=0.0, le=1.0),
):
    """Return pothole detection logs from road_inspection.db."""
    if not ROAD_DB_PATH.exists():
        return {"logs": [], "total": 0, "limit": limit, "offset": offset}

    conn = get_road_db()
    cursor = conn.cursor()

    # Check if potholes table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='potholes'")
    if not cursor.fetchone():
        conn.close()
        return {"logs": [], "total": 0, "limit": limit, "offset": offset}

    cursor.execute(
        """SELECT id, timestamp, confidence, drone_x, drone_y, drone_z, frame
           FROM potholes
           WHERE confidence >= ?
           ORDER BY id DESC
           LIMIT ? OFFSET ?""",
        (min_conf, limit, offset),
    )
    rows = cursor.fetchall()

    cursor.execute(
        "SELECT COUNT(*) FROM potholes WHERE confidence >= ?",
        (min_conf,),
    )
    total = cursor.fetchone()[0]
    conn.close()

    output = []
    for row in rows:
        image_b64 = ""
        if row["frame"]:
            image_b64 = base64.b64encode(row["frame"]).decode("utf-8")

        output.append({
            "id": row["id"],
            "timestamp": row["timestamp"],
            "label": "pothole",
            "confidence": round(row["confidence"], 4),
            "location": {
                "x": round(row["drone_x"], 2),
                "y": round(row["drone_y"], 2),
                "z": round(row["drone_z"], 2),
            },
            "image": f"data:image/jpeg;base64,{image_b64}" if image_b64 else None,
        })

    return {"logs": output, "total": total, "limit": limit, "offset": offset}


@app.get("/pothole-stats")
def get_pothole_stats():
    """Aggregate pothole detection statistics."""
    if not ROAD_DB_PATH.exists():
        return {"total_detections": 0, "avg_confidence": 0, "max_confidence": 0, "min_confidence": 0}

    conn = get_road_db()
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='potholes'")
    if not cursor.fetchone():
        conn.close()
        return {"total_detections": 0, "avg_confidence": 0, "max_confidence": 0, "min_confidence": 0}

    cursor.execute("SELECT COUNT(*) FROM potholes")
    total = cursor.fetchone()[0]

    if total == 0:
        conn.close()
        return {"total_detections": 0, "avg_confidence": 0, "max_confidence": 0, "min_confidence": 0}

    cursor.execute("SELECT AVG(confidence), MAX(confidence), MIN(confidence) FROM potholes")
    avg_c, max_c, min_c = cursor.fetchone()
    conn.close()

    return {
        "total_detections": total,
        "avg_confidence": round(avg_c, 4),
        "max_confidence": round(max_c, 4),
        "min_confidence": round(min_c, 4),
    }


# ─────────────────────────────────────────────
# TRAINING METRICS
# ─────────────────────────────────────────────

@app.get("/training/metrics")
def get_training_metrics():
    """Read results.csv from the latest training run."""
    runs_dir = ROOT_DIR / "runs"
    if not runs_dir.exists():
        return {"error": "No training runs found", "data": []}

    # Find the latest run directory that has results.csv
    csv_path = None
    for run_name in sorted(os.listdir(runs_dir), reverse=True):
        candidate = runs_dir / run_name / "results.csv"
        if candidate.exists():
            csv_path = candidate
            break

    if csv_path is None:
        return {"error": "No results.csv found", "data": []}

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            cleaned = {}
            for k, v in row.items():
                key = k.strip()
                try:
                    cleaned[key] = float(v.strip())
                except (ValueError, AttributeError):
                    cleaned[key] = v.strip() if v else ""
            rows.append(cleaned)

    return {"run_name": csv_path.parent.name, "data": rows}


# ─────────────────────────────────────────────
# TRAINING / EVAL IMAGES
# ─────────────────────────────────────────────

@app.get("/training/images")
def list_training_images():
    """List available eval plot images."""
    results_dir = ROOT_DIR / "results"
    images = []
    for subdir in sorted(results_dir.iterdir()):
        if subdir.is_dir() and "eval" in subdir.name:
            for f in sorted(subdir.iterdir()):
                if f.suffix.lower() in (".png", ".jpg", ".jpeg"):
                    images.append({
                        "name": f.name,
                        "eval_run": subdir.name,
                        "path": f"/{subdir.name}/{f.name}",
                    })
    return {"images": images}


@app.get("/training/images/{eval_run}/{filename}")
def get_training_image(eval_run: str, filename: str):
    """Serve an evaluation image file."""
    img_path = ROOT_DIR / "results" / eval_run / filename
    if not img_path.exists():
        return Response(status_code=404, content="Image not found")
    suffix = img_path.suffix.lower()
    media = "image/png" if suffix == ".png" else "image/jpeg"
    return Response(content=img_path.read_bytes(), media_type=media)


# ─────────────────────────────────────────────
# ON-DEMAND DETECTION
# ─────────────────────────────────────────────

_model = None

def _load_model():
    global _model
    if _model is None:
        from ultralytics import YOLO
        weights = ROOT_DIR / "models" / "best_run9.pt"
        if not weights.exists():
            # fallback to any best_run*.pt
            for f in sorted((ROOT_DIR / "models").glob("best_run*.pt"), reverse=True):
                weights = f
                break
        _model = YOLO(str(weights))
    return _model


@app.post("/detect")
async def detect_image(file: UploadFile = File(...)):
    """Run YOLO inference on an uploaded image."""
    import cv2
    import numpy as np

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "Could not decode image"}

    model = _load_model()
    results = model(img, conf=0.3, verbose=False)

    detections = []
    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            detections.append({
                "class": model.names[cls_id],
                "confidence": round(conf, 4),
                "bbox": box.xyxy[0].tolist(),
            })

    # Get annotated image
    annotated = results[0].plot().copy()
    _, encoded = cv2.imencode(".jpg", annotated)
    annotated_b64 = base64.b64encode(encoded.tobytes()).decode("utf-8")

    return {
        "detections": detections,
        "count": len(detections),
        "annotated_image": f"data:image/jpeg;base64,{annotated_b64}",
    }


# ─────────────────────────────────────────────
# MISSION CONTROL
# ─────────────────────────────────────────────

import subprocess
import signal

_mission_process = None
_mission_start_time = None

FRAME_PATH = ROOT_DIR / "latest_frame.jpg"


@app.get("/mission")
def get_mission_info():
    """Return orbit/mission configuration (read from orbit.py constants)."""
    return {
        "model": "YOLO26m (best_run9.pt)",
        "conf_threshold": 0.3,
        "orbit": {
            "center": {"x": 19.0, "y": 0.0},
            "radius_m": 19.0,
            "altitude_m": 10.0,
            "velocity_mps": 3.0,
        },
        "vision": {
            "target_fps": 18,
            "log_cooldown_s": 2.0,
            "camera": "front_center (Camera 0)",
        },
        "pipeline": [
            "Capture RGB frame via AirSim",
            "Run YOLO26m inference (conf ≥ 0.30)",
            "Extract class, confidence, bounding box",
            "Log to SQLite with NED coordinates",
            "Display live annotated feed",
        ],
    }


@app.get("/mission/status")
def mission_status():
    """Check if the inspection mission is currently running."""
    global _mission_process, _mission_start_time
    if _mission_process is not None:
        poll = _mission_process.poll()
        if poll is None:
            elapsed = time.time() - _mission_start_time if _mission_start_time else 0
            return {"running": True, "elapsed_s": round(elapsed, 1), "pid": _mission_process.pid}
        else:
            _mission_process = None
            _mission_start_time = None
            return {"running": False, "last_exit_code": poll}
    return {"running": False}


@app.post("/mission/start")
def start_mission():
    """Launch orbit.py as a subprocess."""
    global _mission_process, _mission_start_time
    if _mission_process is not None and _mission_process.poll() is None:
        return {"error": "Mission already running", "pid": _mission_process.pid}

    orbit_script = ROOT_DIR / "drone" / "orbit.py"
    if not orbit_script.exists():
        return {"error": f"orbit.py not found at {orbit_script}"}

    # Remove stale frame
    if FRAME_PATH.exists():
        FRAME_PATH.unlink()

    try:
        _mission_process = subprocess.Popen(
            ["python", str(orbit_script)],
            cwd=str(ROOT_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        )
        _mission_start_time = time.time()
        return {"started": True, "pid": _mission_process.pid}
    except Exception as e:
        return {"error": str(e)}


@app.post("/mission/road-inspect/start")
def start_road_inspection():
    """Launch drone/road_inspect.py as a subprocess."""
    global _mission_process, _mission_start_time
    if _mission_process is not None and _mission_process.poll() is None:
        return {"error": "A mission is already running", "pid": _mission_process.pid}

    road_script = ROOT_DIR / "drone" / "road_inspect.py"
    if not road_script.exists():
        return {"error": f"road_inspect.py not found at {road_script}"}

    # Remove stale frame
    if FRAME_PATH.exists():
        FRAME_PATH.unlink()

    try:
        _mission_process = subprocess.Popen(
            ["python", str(road_script)],
            cwd=str(ROOT_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        )
        _mission_start_time = time.time()
        return {"started": True, "pid": _mission_process.pid, "mission": "road_inspect"}
    except Exception as e:
        return {"error": str(e)}


@app.post("/mission/stop")
def stop_mission():
    """Stop the running orbit.py mission."""
    global _mission_process, _mission_start_time
    if _mission_process is None or _mission_process.poll() is not None:
        _mission_process = None
        _mission_start_time = None
        return {"stopped": True, "was_running": False}

    try:
        # On Windows, send CTRL_BREAK to the process group
        _mission_process.send_signal(signal.CTRL_BREAK_EVENT)
        try:
            _mission_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _mission_process.kill()
            _mission_process.wait(timeout=3)
        pid = _mission_process.pid
        _mission_process = None
        _mission_start_time = None
        return {"stopped": True, "was_running": True, "pid": pid}
    except Exception as e:
        _mission_process = None
        _mission_start_time = None
        return {"stopped": True, "error": str(e)}


@app.get("/stream/latest")
def get_latest_frame():
    """Serve the latest annotated frame written by orbit.py."""
    if not FRAME_PATH.exists():
        return Response(status_code=204)  # No content yet

    try:
        frame_bytes = FRAME_PATH.read_bytes()
        return Response(content=frame_bytes, media_type="image/jpeg")
    except Exception:
        return Response(status_code=204)