from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class RouteRequest(BaseModel):
    polyline: str
    sample_meters: int = 30

@app.post("/scan_route")
def scan_route(req: RouteRequest):
    return {
        "hazards": [
            {"lat": 40.7123, "lng": -74.0061, "type": "pothole", "conf": 0.82}
        ]
    }
