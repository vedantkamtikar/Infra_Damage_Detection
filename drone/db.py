"""
db.py

SQLite database for logging crack detections during drone inspection.

Schema:
    detections (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp   TEXT,
        frame_id    INTEGER,
        ned_x       REAL,
        ned_y       REAL,
        ned_z       REAL,
        class_name  TEXT,
        confidence  REAL,
        bbox_x      REAL,
        bbox_y      REAL,
        bbox_w      REAL,
        bbox_h      REAL,
        image_path  TEXT
    )
"""

import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "results" / "detections.db"


class DetectionDB:

    def __init__(self, db_path: Path = DB_PATH):
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path))
        self._create_table()
        print(f"  ✓ Database connected: {db_path}")

    def _create_table(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS detections (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp   TEXT,
                frame_id    INTEGER,
                ned_x       REAL,
                ned_y       REAL,
                ned_z       REAL,
                class_name  TEXT,
                confidence  REAL,
                bbox_x      REAL,
                bbox_y      REAL,
                bbox_w      REAL,
                bbox_h      REAL,
                image_path  TEXT
            )
        """)
        self.conn.commit()

    def log(self, frame_id: int, position: tuple, detections: list, image_path: str):
        """Log all detections from a single frame."""
        ned_x, ned_y, ned_z = position
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        for det in detections:
            self.conn.execute("""
                INSERT INTO detections
                (timestamp, frame_id, ned_x, ned_y, ned_z,
                 class_name, confidence, bbox_x, bbox_y, bbox_w, bbox_h, image_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                timestamp,
                frame_id,
                ned_x, ned_y, ned_z,
                det["class_name"],
                det["confidence"],
                det["bbox_x"],
                det["bbox_y"],
                det["bbox_w"],
                det["bbox_h"],
                image_path,
            ))
        self.conn.commit()

    def summary(self):
        """Print a summary of all detections in the database."""
        cursor = self.conn.execute("SELECT COUNT(*) FROM detections")
        total = cursor.fetchone()[0]

        print("\n" + "─" * 45)
        print("  Detection Summary")
        print("─" * 45)
        print(f"  Total detections : {total}")

        if total == 0:
            print("  No cracks detected during this mission.")
            print("─" * 45)
            return

        # Confidence stats
        cursor = self.conn.execute("""
            SELECT MIN(confidence), MAX(confidence), AVG(confidence)
            FROM detections
        """)
        min_c, max_c, avg_c = cursor.fetchone()
        print(f"  Confidence min   : {min_c:.3f}")
        print(f"  Confidence max   : {max_c:.3f}")
        print(f"  Confidence avg   : {avg_c:.3f}")

        # Frames with detections
        cursor = self.conn.execute("""
            SELECT COUNT(DISTINCT frame_id) FROM detections
        """)
        frames_with_dets = cursor.fetchone()[0]
        print(f"  Frames with dets : {frames_with_dets}")

        # Top 5 highest confidence detections
        print(f"\n  Top detections:")
        cursor = self.conn.execute("""
            SELECT frame_id, ned_x, ned_y, ned_z, confidence, image_path
            FROM detections
            ORDER BY confidence DESC
            LIMIT 5
        """)
        for row in cursor.fetchall():
            frame_id, x, y, z, conf, path = row
            print(f"    Frame {frame_id:04d} | "
                  f"NED ({x:.1f}, {y:.1f}, {z:.1f}) | "
                  f"conf {conf:.3f}")

        print("─" * 45)

    def clear(self):
        """Clear all records from the database."""
        self.conn.execute("DELETE FROM detections")
        self.conn.commit()
        print("  ✓ Database cleared")

    def close(self):
        """Close the database connection."""
        self.conn.close()