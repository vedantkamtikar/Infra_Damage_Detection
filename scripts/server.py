from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import sqlite3
import base64

app = FastAPI()

# Allow React to talk to this API
app.add_middleware(CORSMiddleware, allow_origins=["*"])

@app.get("/logs")
def get_logs():
    conn = sqlite3.connect("detections.db")
    cursor = conn.cursor()
    # Get the last 10 detections
    cursor.execute("SELECT id, timestamp, class, confidence, x, y, z, frame FROM detections ORDER BY id DESC LIMIT 10")
    rows = cursor.fetchall()
    conn.close()

    output = []
    for row in rows:
        # Convert binary BLOB to Base64 string for React
        image_base64 = ""
        if row[7]:
            image_base64 = base64.b64encode(row[7]).decode('utf-8')
            
        output.append({
            "id": row[0],
            "timestamp": row[1],
            "label": row[2],
            "confidence": row[3],
            "location": {"x": row[4], "y": row[5], "z": row[6]},
            "image": f"data:image/jpeg;base64,{image_base64}"
        })
    return output