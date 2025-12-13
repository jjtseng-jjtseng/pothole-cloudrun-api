import os
import math
from typing import List, Dict, Any, Optional

import requests
import polyline
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# YOLO (Ultralytics)
from ultralytics import YOLO
import numpy as np
import cv2

app = FastAPI()

# --- Config via env vars (set these in Cloud Run) ---
GOOGLE_MAPS_KEY = os.environ.get("GOOGLE_MAPS_KEY", "")
MODEL_PATH = os.environ.get("MODEL_PATH", "best.pt")  # put best.pt in repo root or download it at startup
CONF_THRESH = float(os.environ.get("CONF_THRESH", "0.35"))

# Keep this low at first so it doesn't time out
MAX_POINTS = int(os.environ.get("MAX_POINTS", "60"))          # max Street View images per request
SAMPLE_EVERY_N = int(os.environ.get("SAMPLE_EVERY_N", "3"))   # take every Nth route point
STREETVIEW_SIZE = os.environ.get("STREETVIEW_SIZE", "640x640")
FOV = int(os.environ.get("STREETVIEW_FOV", "90"))
PITCH = int(os.environ.get("STREETVIEW_PITCH", "-15"))

if not GOOGLE_MAPS_KEY:
    print("WARNING: GOOGLE_MAPS_KEY is not set.")

# Load YOLO once at startup (important for speed)
model: Optional[YOLO] = None

@app.on_event("startup")
def _startup():
    global model
    try:
        model = YOLO(MODEL_PATH)
        print(f"Loaded model from {MODEL_PATH}")
    except Exception as e:
        print(f"ERROR loading model: {e}")
        model = None


class RoutePutRequest(BaseModel):
    Curr_Lat: float
    Curr_Lng: float
    Dest_Lat: float
    Dest_Lng: float


@app.get("/")
def root():
    return {"status": "ok", "message": "Pothole API running. Use /docs"}


def calculate_heading(lat1, lon1, lat2, lon2) -> float:
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1) * math.cos(lat2) * math.cos(dlon))
    bearing = math.degrees(math.atan2(x, y))
    return (bearing + 360) % 360


def get_route_points(origin_lat, origin_lng, dest_lat, dest_lng) -> List[List[float]]:
    if not GOOGLE_MAPS_KEY:
        raise HTTPException(status_code=500, detail="Server missing GOOGLE_MAPS_KEY")

    url = (
        "https://maps.googleapis.com/maps/api/directions/json"
        f"?origin={origin_lat},{origin_lng}"
        f"&destination={dest_lat},{dest_lng}"
        f"&mode=driving"
        f"&key={GOOGLE_MAPS_KEY}"
    )
    r = requests.get(url, timeout=20)
    data = r.json()

    if data.get("status") != "OK":
        raise HTTPException(status_code=400, detail=f"Directions API error: {data.get('status')}")

    route = data["routes"][0]
    polyline_points = route["overview_polyline"]["points"]
    coords = polyline.decode(polyline_points)  # list of (lat, lng)
    return [[float(a), float(b)] for a, b in coords]


def fetch_streetview_image(lat, lng, heading) -> np.ndarray:
    # Street View Static API (image bytes)
    url = (
        "https://maps.googleapis.com/maps/api/streetview"
        f"?size={STREETVIEW_SIZE}"
        f"&location={lat},{lng}"
        f"&fov={FOV}"
        f"&heading={heading}"
        f"&pitch={PITCH}"
        f"&key={GOOGLE_MAPS_KEY}"
    )
    r = requests.get(url, timeout=25)
    if r.status_code != 200:
        raise RuntimeError(f"Street View HTTP {r.status_code}")

    # Decode JPEG bytes into image array (BGR)
    img_arr = np.frombuffer(r.content, dtype=np.uint8)
    img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("Failed to decode image")
    return img


def run_yolo(img_bgr: np.ndarray) -> List[Dict[str, Any]]:
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded on server")

    # Ultralytics can accept numpy arrays
    results = model.predict(source=img_bgr, conf=CONF_THRESH, verbose=False)
    out = []

    for res in results:
        names = res.names  # class id -> name
        if res.boxes is None:
            continue
        for b in res.boxes:
            cls = int(b.cls[0].item())
            conf = float(b.conf[0].item())
            out.append({
                "class_id": cls,
                "class_name": names.get(cls, str(cls)),
                "conf": conf
            })
    return out


@app.put("/scan_route")
def scan_route(req: RoutePutRequest):
    # 1) Get route points
    coords = get_route_points(req.Curr_Lat, req.Curr_Lng, req.Dest_Lat, req.Dest_Lng)

    if len(coords) < 2:
        return {"hazards": [], "detail": "Route too short"}

    # 2) Sample points so it doesn't explode in time/cost
    sampled = coords[::max(1, SAMPLE_EVERY_N)]
    if len(sampled) > MAX_POINTS:
        sampled = sampled[:MAX_POINTS]

    hazards = []
    scanned = 0

    # 3) For each sampled point, compute heading using next point (if possible)
    for i in range(len(sampled)):
        lat1, lon1 = sampled[i]
        if i < len(sampled) - 1:
            lat2, lon2 = sampled[i + 1]
        else:
            lat2, lon2 = sampled[i - 1]  # fallback

        heading = calculate_heading(lat1, lon1, lat2, lon2)

        try:
            img = fetch_streetview_image(lat1, lon1, heading)
            dets = run_yolo(img)
            scanned += 1

            # If any detections, record this coordinate as a hazard point
            for d in dets:
                hazards.append({
                    "lat": lat1,
                    "lng": lon1,
                    "type": d["class_name"],   # you can map this to "pothole"/"manhole" etc
                    "conf": d["conf"]
                })

        except Exception as e:
            # skip bad points but continue
            continue

    return {
        "hazards": hazards,
        "meta": {
            "route_points": len(coords),
            "sampled_points": len(sampled),
            "scanned_images": scanned
        }
    }
