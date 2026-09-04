"""
simulate_bus.py
Standalone bus simulator — run separately from the FastAPI server.
Simulates 3 buses on Bengaluru routes, POSTing fake detections every 10s.

Run: python simulate_bus.py
Requires backend running at http://localhost:8000
"""
import random
import time
import json
from datetime import datetime

import httpx

BACKEND_URL = "http://localhost:8000"
INTERVAL_SECONDS = 10

BUSES = {
    "Bus 12": {"lat": 12.9716, "lng": 77.5946, "route": "Silk Board → Koramangala"},
    "Bus 7":  {"lat": 12.9352, "lng": 77.6245, "route": "Electronic City → HSR Layout"},
    "Bus 23": {"lat": 13.0100, "lng": 77.5500, "route": "Yeshwantpur → Rajajinagar"},
}

DEFECT_TYPES = ["Pothole", "Longitudinal Crack", "Transverse Crack", "Alligator Crack"]

BENGALURU_CENTER = (12.9716, 77.5946)
ROUTE_RADIUS = 0.05   # ~5.5km in degrees


def clamp(val, lo, hi):
    return max(lo, min(hi, val))


def move_bus(bus: dict) -> dict:
    """Randomly walk the bus position within Bengaluru bounds."""
    bus["lat"] = clamp(
        bus["lat"] + random.uniform(-0.003, 0.003),
        BENGALURU_CENTER[0] - ROUTE_RADIUS,
        BENGALURU_CENTER[0] + ROUTE_RADIUS,
    )
    bus["lng"] = clamp(
        bus["lng"] + random.uniform(-0.003, 0.003),
        BENGALURU_CENTER[1] - ROUTE_RADIUS,
        BENGALURU_CENTER[1] + ROUTE_RADIUS,
    )
    return bus


def main():
    print("=" * 55)
    print("  Road Defect Bus Simulator — Bengaluru")
    print(f"  Backend: {BACKEND_URL}")
    print(f"  Buses: {', '.join(BUSES.keys())}")
    print(f"  Interval: {INTERVAL_SECONDS}s")
    print("=" * 55)

    # Verify backend is up
    try:
        r = httpx.get(f"{BACKEND_URL}/health", timeout=5)
        print(f"Backend health: {r.json()['status']}\n")
    except Exception as e:
        print(f"WARNING: Backend not reachable — {e}")
        print("Make sure uvicorn main:app --reload is running.\n")

    iteration = 0
    while True:
        iteration += 1
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"\n[{ts}] Tick #{iteration}")

        for bus_id, bus in BUSES.items():
            move_bus(bus)

            has_detection = random.random() < 0.65   # 65% chance
            defect_type = random.choice(DEFECT_TYPES) if has_detection else None
            confidence = round(random.uniform(0.55, 0.97), 2) if has_detection else None

            payload = {
                "bus_id": bus_id,
                "lat": round(bus["lat"], 6),
                "lng": round(bus["lng"], 6),
                "defect_type": defect_type,
                "confidence": confidence,
            }

            try:
                r = httpx.post(
                    f"{BACKEND_URL}/bus-ping",
                    json=payload,
                    timeout=5,
                )
                status_icon = "✓" if r.status_code == 200 else "✗"
                if defect_type:
                    print(f"  {status_icon} {bus_id:<8} {defect_type:<22} {int(confidence*100)}%  "
                          f"@ {payload['lat']:.4f}, {payload['lng']:.4f}")
                else:
                    print(f"  {status_icon} {bus_id:<8} (no defect)                    "
                          f"@ {payload['lat']:.4f}, {payload['lng']:.4f}")
            except Exception as e:
                print(f"  ✗ {bus_id}: POST failed — {e}")

        print(f"  Sleeping {INTERVAL_SECONDS}s...")
        time.sleep(INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
