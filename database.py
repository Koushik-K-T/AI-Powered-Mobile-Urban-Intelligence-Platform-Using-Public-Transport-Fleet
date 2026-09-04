import aiosqlite
import math
import os
from datetime import datetime, timezone

DB_PATH = "road_defects.db"


def haversine(lat1, lon1, lat2, lon2) -> float:
    """Returns distance in metres between two GPS coords."""
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def defect_priority(defect_type: str) -> str:
    if defect_type in ("Pothole", "Alligator Crack"):
        return "High"
    if defect_type == "Transverse Crack":
        return "Medium"
    return "Low"


async def init_db():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS detections (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id      TEXT NOT NULL,
                source      TEXT NOT NULL,
                bus_id      TEXT,
                frame_num   INTEGER,
                defect_type TEXT NOT NULL,
                confidence  REAL NOT NULL,
                lat         REAL NOT NULL,
                lng         REAL NOT NULL,
                thumbnail   TEXT,
                created_at  TEXT NOT NULL
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS issues (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                cluster_key     TEXT UNIQUE NOT NULL,
                defect_type     TEXT NOT NULL,
                lat             REAL NOT NULL,
                lng             REAL NOT NULL,
                detection_count INTEGER DEFAULT 1,
                priority        TEXT NOT NULL,
                status          TEXT DEFAULT 'Pending',
                confirmed       INTEGER DEFAULT 0,
                first_seen      TEXT NOT NULL,
                last_seen       TEXT NOT NULL
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS bus_pings (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                bus_id      TEXT NOT NULL,
                lat         REAL NOT NULL,
                lng         REAL NOT NULL,
                defect_type TEXT,
                confidence  REAL,
                created_at  TEXT NOT NULL
            )
        """)
        await db.commit()


async def insert_detection(job_id, source, bus_id, frame_num, defect_type,
                           confidence, lat, lng, thumbnail=None) -> int:
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            """INSERT INTO detections
               (job_id, source, bus_id, frame_num, defect_type, confidence,
                lat, lng, thumbnail, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (job_id, source, bus_id, frame_num, defect_type, confidence,
             lat, lng, thumbnail, now_iso())
        )
        await db.commit()
        return cursor.lastrowid


async def get_detections_by_job(job_id: str) -> list:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM detections WHERE job_id=? ORDER BY frame_num ASC", (job_id,)
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]


async def get_live_feed(limit: int = 50) -> list:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            """SELECT bus_id, defect_type, confidence, lat, lng, created_at, 'detection' as kind
               FROM detections WHERE source='bus' AND defect_type IS NOT NULL
               UNION ALL
               SELECT bus_id, defect_type, confidence, lat, lng, created_at, 'ping' as kind
               FROM bus_pings
               ORDER BY created_at DESC LIMIT ?""",
            (limit,)
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]


async def insert_bus_ping(bus_id, lat, lng, defect_type=None, confidence=None):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO bus_pings (bus_id, lat, lng, defect_type, confidence, created_at) VALUES (?,?,?,?,?,?)",
            (bus_id, lat, lng, defect_type, confidence, now_iso())
        )
        await db.commit()


async def get_issues() -> list:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM issues ORDER BY detection_count DESC")
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]


async def cluster_and_fuse(lat: float, lng: float, defect_type: str):
    """Group nearby detections into issues. 2+ within 20m = Confirmed."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM issues WHERE defect_type=?", (defect_type,)
        )
        existing = await cursor.fetchall()

        matched = None
        for row in existing:
            d = haversine(lat, lng, row["lat"], row["lng"])
            if d < 20:
                matched = dict(row)
                break

        ts = now_iso()
        if matched:
            new_count = matched["detection_count"] + 1
            confirmed = 1 if new_count >= 2 else 0
            await db.execute(
                """UPDATE issues SET detection_count=?, confirmed=?, last_seen=?
                   WHERE id=?""",
                (new_count, confirmed, ts, matched["id"])
            )
        else:
            cluster_key = f"{round(lat,4)}_{round(lng,4)}_{defect_type}"
            await db.execute(
                """INSERT OR IGNORE INTO issues
                   (cluster_key, defect_type, lat, lng, detection_count,
                    priority, status, confirmed, first_seen, last_seen)
                   VALUES (?,?,?,?,1,?,?,0,?,?)""",
                (cluster_key, defect_type, lat, lng,
                 defect_priority(defect_type), "Pending", ts, ts)
            )
        await db.commit()
