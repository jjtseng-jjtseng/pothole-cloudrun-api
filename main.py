# main.py
import os
import math
import uuid
import tempfile
from typing import List, Optional, Dict, Any, Tuple
from urllib.parse import quote

import requests
import polyline as polyline_lib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# NOTE: ultralytics imports cv2 internally
from ultralytics import YOLO


# ----------------------------
# Config
# ----------------------------
GOOGLE_MAPS_KEY = os.environ.get("GOOGLE_MAPS_KEY", "")
MODEL_PATH = os.environ.get("MODEL_PATH", "best.pt")

# Safety/perf limits (so one request can't explode your bill)
DEFAULT_SAMPLE_METERS = 30
MAX_POINTS = 200          # hard cap of sampled points
STREETVIEW_SIZE = "640x640"
STREETVIEW_FOV = 90
STREETVIEW_PITCH = -10
CONF_THRES = 0.45         # tweak as needed

# Firebase RTDB (public in your case)
# IMPORTANT: default to "" so .rstrip doesn't crash if env var is missing
FIREBASE_DB_URL = os.environ.get("FIREBASE_DB_URL", "").rstrip("/")
FIREBASE_ROOT_NODE = os.environ.get("FIREBASE_ROOT_NODE", "detectRoundChunk")

# Only upload detections >= 0.60 confidence
FIREBASE_MIN_CONF = float(os.environ.get("FIREBASE_MIN_CONF", "0.60"))

# Formatting to match your screenshot
LOCATION_DECIMALS = int(os.environ.get("LOCATION_DECIMALS", "5"))  # "40.40988, -74.54944"
CHUNK_DECIMALS = int(os.environ.get("CHUNK_DECIMALS", "1"))        # "40@4"

# Timeout for Firebase REST calls
FIREBASE_TIMEOUT_S = float(os.environ.get("FIREBASE_TIMEOUT_S", "10"))


# ----------------------------
# App + model
# ----------------------------
app = FastAPI(title="Pothole Route Scanner API", version="1.0.0")

_model: Optional[YOLO] = None


def get_model() -> YOLO:
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError(
                f"MODEL_PATH not found: '{MODEL_PATH}'. "
                "Make sure best.pt is in the container and MODEL_PATH points to it."
            )
        _model = YOLO(MODEL_PATH)
    return _model


# ----------------------------
# Request/response schemas
# ----------------------------
class RouteScanRequest(BaseModel):
    # Option A: send start/end
    start_lat: Optional[float] = None
    start_lng: Optional[float] = None
    end_lat: Optional[float] = None
    end_lng: Optional[float] = None

    # Option B: send a polyline directly (Google encoded polyline string)
    polyline: Optional[str] = None

    # Sampling distance along the route
    sample_meters: int = Field(default=DEFAULT_SAMPLE_METERS, ge=5, le=200)

    # Optional knobs
    max_points: int = Field(default=MAX_POINTS, ge=10, le=MAX_POINTS)
    confidence: float = Field(default=CONF_THRES, ge=0.01, le=0.99)

    # (Kept for compatibility/future use, but Firebase uploads will use Contributor="AI")
    contributor: str = Field(default="Unknown", max_length=80)


class PotholeHit(BaseModel):
    lat: float
    lng: float
    confidence: float
    cls: int
    class_name: str
    heading: Optional[float] = None
    image_id: Optional[str] = None


class RouteScanResponse(BaseModel):
    count: int
    potholes: List[PotholeHit]
    route_points_sampled: int
    sampled_points: List[Dict[str, float]]  # list of {"lat":..., "lng":...} so MIT can show “waypoints”


# ----------------------------
# Helpers
# ----------------------------
def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371000.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compass bearing 0..360 from (lat1,lon1) to (lat2,lon2)."""
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    brng = math.degrees(math.atan2(y, x))
    return (brng + 360.0) % 360.0


def sample_route_points(coords: List[Tuple[float, float]], every_m: int, max_points: int) -> List[Tuple[float, float]]:
    """Downsample polyline coords to ~every_m meters."""
    if not coords:
        return []
    sampled = [coords[0]]
    accum = 0.0
    for i in range(1, len(coords)):
        if len(sampled) >= max_points:
            break
        lat1, lon1 = coords[i - 1]
        lat2, lon2 = coords[i]
        d = haversine_m(lat1, lon1, lat2, lon2)
        accum += d
        if accum >= every_m:
            sampled.append((lat2, lon2))
            accum = 0.0
    if len(sampled) < max_points and coords[-1] != sampled[-1]:
        sampled.append(coords[-1])
    return sampled[:max_points]


def get_polyline_from_google(start_lat: float, start_lng: float, end_lat: float, end_lng: float) -> str:
    if not GOOGLE_MAPS_KEY:
        raise HTTPException(status_code=500, detail="GOOGLE_MAPS_KEY env var not set on Cloud Run.")
    url = "https://maps.googleapis.com/maps/api/directions/json"
    params = {
        "origin": f"{start_lat},{start_lng}",
        "destination": f"{end_lat},{end_lng}",
        "mode": "driving",
        "key": GOOGLE_MAPS_KEY,
    }
    r = requests.get(url, params=params, timeout=25)
    data = r.json()
    if data.get("status") != "OK":
        raise HTTPException(
            status_code=400,
            detail=f"Directions API error: {data.get('status')} {data.get('error_message','')}".strip()
        )
    return data["routes"][0]["overview_polyline"]["points"]


def download_streetview_image(lat: float, lng: float, heading: Optional[float], out_path: str) -> None:
    if not GOOGLE_MAPS_KEY:
        raise HTTPException(status_code=500, detail="GOOGLE_MAPS_KEY env var not set on Cloud Run.")
    url = "https://maps.googleapis.com/maps/api/streetview"
    params = {
        "size": STREETVIEW_SIZE,
        "location": f"{lat},{lng}",
        "fov": str(STREETVIEW_FOV),
        "pitch": str(STREETVIEW_PITCH),
        "key": GOOGLE_MAPS_KEY,
    }
    if heading is not None:
        params["heading"] = str(heading)

    resp = requests.get(url, params=params, timeout=30)
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Street View download failed HTTP {resp.status_code}")
    # If Google returns an error image sometimes it’s still 200; keep it simple for now.
    with open(out_path, "wb") as f:
        f.write(resp.content)


# ----------------------------
# Firebase helpers (RTDB REST)
# ----------------------------
def _chunk_token(x: float, decimals: int = 1) -> str:
    # round to 1 decimal -> 40.4 ; replace '.' with '@' -> "40@4"
    s = f"{round(float(x), decimals):.{decimals}f}"
    return s.replace(".", "@")


def _chunk_key(lat: float, lng: float) -> str:
    # EXACT formatting: "40@4, -74@5" (comma + space)
    return f"{_chunk_token(lat, CHUNK_DECIMALS)}, {_chunk_token(lng, CHUNK_DECIMALS)}"


def _location_string(lat: float, lng: float) -> str:
    # EXACT formatting: "40.40988, -74.54944"
    return f"{lat:.{LOCATION_DECIMALS}f}, {lng:.{LOCATION_DECIMALS}f}"


def _fb_url(*parts: str) -> str:
    # Encode each path segment safely for Firebase REST URLs
    encoded = "/".join(quote(p, safe="") for p in parts)
    return f"{FIREBASE_DB_URL}/{encoded}.json"


def _firebase_get(path_parts: List[str]) -> Any:
    url = _fb_url(*path_parts)
    r = requests.get(url, timeout=FIREBASE_TIMEOUT_S)
    r.raise_for_status()
    return r.json()


def _firebase_put(path_parts: List[str], payload: Dict[str, Any]) -> None:
    url = _fb_url(*path_parts)
    r = requests.put(url, json=payload, timeout=FIREBASE_TIMEOUT_S)
    r.raise_for_status()


def upload_detection_to_firebase(*, lat: float, lng: float, class_name: str, confidence: float) -> None:
    """
    Writes to:
      detectRoundChunk/<chunkKey>/<index> = {
        Classes, Contributor, Location, ReportCount, Confidence
      }

    - Always APPENDS a new numeric index (no merging).
    - Fixes the “keeps writing under 0” bug by handling dict OR list JSON returns.
    """
    if not FIREBASE_DB_URL:
        # Fail fast so you see it in logs if env var isn't set
        raise RuntimeError("FIREBASE_DB_URL env var not set.")

    chunk = _chunk_key(lat, lng)
    location = _location_string(lat, lng)
    class_norm = str(class_name).lower().strip()

    chunk_path = [FIREBASE_ROOT_NODE, chunk]

    existing = _firebase_get(chunk_path)

    # Compute next numeric index robustly
    next_index_int = 0

    if existing is None:
        next_index_int = 0

    elif isinstance(existing, dict):
        numeric_keys: List[int] = []
        for k in existing.keys():
            try:
                numeric_keys.append(int(k))
            except Exception:
                pass
        next_index_int = (max(numeric_keys) + 1) if numeric_keys else 0

    elif isinstance(existing, list):
        # Firebase can return list when keys are 0..N.
        # Append after last non-null element.
        last_used = -1
        for i, item in enumerate(existing):
            if item is not None:
                last_used = i
        next_index_int = last_used + 1  # if all None -> 0

    else:
        # Unexpected type; avoid crash
        next_index_int = 0

    payload = {
        "Classes": class_norm,
        "Contributor": "AI",
        "Location": location,
        "ReportCount": 1,
        "Confidence": float(round(float(confidence), 3)),
    }

    _firebase_put(chunk_path + [str(next_index_int)], payload)


# ----------------------------
# Routes
# ----------------------------
@app.get("/")
def root():
    return {
        "ok": True,
        "message": "Pothole API running. Use POST /scan_route",
        "endpoints": ["/docs", "/openapi.json", "/scan_route"],
    }


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.post("/scan_route", response_model=RouteScanResponse)
def scan_route(req: RouteScanRequest):
    # 1) Get polyline
    poly = req.polyline
    if not poly:
        # require start/end
        if None in (req.start_lat, req.start_lng, req.end_lat, req.end_lng):
            raise HTTPException(
                status_code=422,
                detail="Provide either 'polyline' OR all of: start_lat, start_lng, end_lat, end_lng.",
            )
        poly = get_polyline_from_google(req.start_lat, req.start_lng, req.end_lat, req.end_lng)

    # 2) Decode to coords and sample
    coords = polyline_lib.decode(poly)  # [(lat,lng), ...]
    if len(coords) < 2:
        raise HTTPException(status_code=400, detail="Polyline decoded to fewer than 2 points.")

    sampled = sample_route_points(coords, every_m=req.sample_meters, max_points=req.max_points)
    if len(sampled) < 1:
        raise HTTPException(status_code=400, detail="No sampled points generated.")

    # 3) Run Street View + YOLO, mapping detections -> the (lat,lng) where the image was taken
    model = get_model()
    potholes: List[PotholeHit] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for i, (lat, lng) in enumerate(sampled):
            # heading toward next point (optional but helps consistency)
            heading = None
            if i < len(sampled) - 1:
                lat2, lng2 = sampled[i + 1]
                heading = bearing_deg(lat, lng, lat2, lng2)

            image_id = f"{i}_{uuid.uuid4().hex[:8]}"
            img_path = os.path.join(tmpdir, f"sv_{image_id}.jpg")

            download_streetview_image(lat, lng, heading, img_path)

            # Ultralytics predict
            results = model.predict(source=img_path, conf=req.confidence, verbose=False)

            # If any box appears, treat that coordinate as a hazard “waypoint”
            for r in results:
                names = r.names or {}
                if r.boxes is None or len(r.boxes) == 0:
                    continue
                for b in r.boxes:
                    conf = float(b.conf.item()) if hasattr(b.conf, "item") else float(b.conf)
                    cls = int(b.cls.item()) if hasattr(b.cls, "item") else int(b.cls)
                    class_name = str(names.get(cls, cls))

                    potholes.append(
                        PotholeHit(
                            lat=float(lat),
                            lng=float(lng),
                            confidence=conf,
                            cls=cls,
                            class_name=class_name,
                            heading=float(heading) if heading is not None else None,
                            image_id=image_id,
                        )
                    )

                    # ✅ Firebase upload only if >= FIREBASE_MIN_CONF confidence
                    if conf >= FIREBASE_MIN_CONF:
                        try:
                            upload_detection_to_firebase(
                                lat=float(lat),
                                lng=float(lng),
                                class_name=class_name,
                                confidence=conf,
                            )
                        except Exception as e:
                            # Extra change: log the error instead of silently swallowing it
                            print("Firebase upload failed:", repr(e))

    # 4) Return: pothole “waypoints” + the sampled route points (so MIT can compare along navigation)
    sampled_points_payload = [{"lat": float(a), "lng": float(b)} for (a, b) in sampled]

    return RouteScanResponse(
        count=len(potholes),
        potholes=potholes,
        route_points_sampled=len(sampled),
        sampled_points=sampled_points_payload,
    )
