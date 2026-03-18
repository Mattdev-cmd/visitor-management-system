"""
Visitor CRUD service — bridges the web layer and the database.
"""

import os
import pickle
import uuid
from datetime import datetime
from typing import Optional

import cv2
import numpy as np

import config
from database import get_db
from services.face_service import encode_face, serialize_encoding


# ---------------------------------------------------------------------- #
#  Helpers
# ---------------------------------------------------------------------- #

def _save_photo(frame_bgr: np.ndarray, prefix: str = "visitor") -> str:
    """Write *frame_bgr* to the uploads folder and return the relative path."""
    filename = f"{prefix}_{uuid.uuid4().hex[:12]}.jpg"
    abs_path = os.path.join(config.UPLOAD_FOLDER, filename)
    cv2.imwrite(abs_path, frame_bgr)
    return f"uploads/{filename}"


# ---------------------------------------------------------------------- #
#  Registration
# ---------------------------------------------------------------------- #

def register_visitor(
    name: str,
    purpose: str,
    contact: str,
    photo_frame: Optional[np.ndarray] = None,
) -> int:
    """Create a new visitor record and return the visitor ID."""
    photo_path = None
    face_blob = None

    if photo_frame is not None:
        photo_path = _save_photo(photo_frame, prefix="reg")
        encoding = encode_face(photo_frame)
        if encoding is not None:
            face_blob = serialize_encoding(encoding)

    with get_db() as conn:
        cur = conn.execute(
            """INSERT INTO visitors (name, purpose, contact, photo_path, face_encoding)
               VALUES (?, ?, ?, ?, ?)""",
            (name, purpose, contact, photo_path, face_blob),
        )
        return cur.lastrowid


# ---------------------------------------------------------------------- #
#  Visit logging
# ---------------------------------------------------------------------- #

def log_visit(
    visitor_id: int,
    emotion: str = "Neutral",
    confidence: float = 0.0,
    photo_frame: Optional[np.ndarray] = None,
) -> int:
    """Create a visit log entry and return its ID."""
    photo_path = None
    if photo_frame is not None:
        photo_path = _save_photo(photo_frame, prefix="visit")

    with get_db() as conn:
        cur = conn.execute(
            """INSERT INTO visit_logs (visitor_id, emotion, confidence, photo_path)
               VALUES (?, ?, ?, ?)""",
            (visitor_id, emotion, confidence, photo_path),
        )
        return cur.lastrowid


def checkout_visitor(log_id: int):
    """Set checked_out_at on a visit log."""
    with get_db() as conn:
        conn.execute(
            "UPDATE visit_logs SET checked_out_at = datetime('now','localtime') WHERE id = ?",
            (log_id,),
        )


# ---------------------------------------------------------------------- #
#  Queries
# ---------------------------------------------------------------------- #

def get_visitor(visitor_id: int) -> Optional[dict]:
    with get_db() as conn:
        row = conn.execute("SELECT * FROM visitors WHERE id = ?", (visitor_id,)).fetchone()
    return dict(row) if row else None


def get_all_visitors(limit: int = 200):
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM visitors ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def search_visitors(query: str, limit: int = 50):
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM visitors WHERE name LIKE ? ORDER BY created_at DESC LIMIT ?",
            (f"%{query}%", limit),
        ).fetchall()
    return [dict(r) for r in rows]


def get_visit_logs(visitor_id: Optional[int] = None, limit: int = 200):
    with get_db() as conn:
        if visitor_id:
            rows = conn.execute(
                """SELECT vl.*, v.name AS visitor_name
                   FROM visit_logs vl
                   JOIN visitors v ON vl.visitor_id = v.id
                   WHERE vl.visitor_id = ?
                   ORDER BY vl.checked_in_at DESC LIMIT ?""",
                (visitor_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT vl.*, v.name AS visitor_name
                   FROM visit_logs vl
                   JOIN visitors v ON vl.visitor_id = v.id
                   ORDER BY vl.checked_in_at DESC LIMIT ?""",
                (limit,),
            ).fetchall()
    return [dict(r) for r in rows]


def get_dashboard_stats() -> dict:
    """Aggregate numbers for the dashboard."""
    with get_db() as conn:
        total_visitors = conn.execute("SELECT COUNT(*) FROM visitors").fetchone()[0]
        today = datetime.now().strftime("%Y-%m-%d")
        visits_today = conn.execute(
            "SELECT COUNT(*) FROM visit_logs WHERE checked_in_at LIKE ?",
            (f"{today}%",),
        ).fetchone()[0]
        active_visits = conn.execute(
            "SELECT COUNT(*) FROM visit_logs WHERE checked_out_at IS NULL"
        ).fetchone()[0]
        emotion_counts = conn.execute(
            """SELECT emotion, COUNT(*) as cnt
               FROM visit_logs
               WHERE checked_in_at LIKE ?
               GROUP BY emotion ORDER BY cnt DESC""",
            (f"{today}%",),
        ).fetchall()
    return {
        "total_visitors": total_visitors,
        "visits_today": visits_today,
        "active_visits": active_visits,
        "emotions_today": {r["emotion"]: r["cnt"] for r in emotion_counts},
    }


def get_daily_emotion_trends(days: int = 7) -> dict:
    """Get daily emotion counts for the past N days.
    
    Returns:
        Dict with dates as keys and emotion counts as values.
        Example: {
            "2026-03-18": {"Angry": 2, "Happy": 5, "Neutral": 3},
            "2026-03-17": {"Angry": 1, "Happy": 4, "Neutral": 6}
        }
    """
    from datetime import timedelta
    
    result = {}
    with get_db() as conn:
        for i in range(days):
            date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
            emotion_counts = conn.execute(
                """SELECT emotion, COUNT(*) as cnt
                   FROM visit_logs
                   WHERE DATE(checked_in_at) = ?
                   GROUP BY emotion ORDER BY emotion""",
                (date,),
            ).fetchall()
            result[date] = {r["emotion"]: r["cnt"] for r in emotion_counts}
    
    # Return in chronological order (oldest first)
    return {k: result[k] for k in sorted(result.keys())}


def get_weekly_emotion_trends(weeks: int = 4) -> dict:
    """Get weekly emotion counts for the past N weeks.
    
    Returns:
        Dict with week identifiers as keys and emotion counts as values.
        Example: {
            "Week 12": {"Angry": 10, "Happy": 25, "Neutral": 15},
            "Week 11": {"Angry": 8, "Happy": 22, "Neutral": 18}
        }
    """
    from datetime import timedelta
    
    result = {}
    with get_db() as conn:
        for i in range(weeks):
            # Calculate week start (Monday)
            today = datetime.now()
            week_start = today - timedelta(days=today.weekday() + (7 * i))
            week_num = week_start.isocalendar()[1]
            week_key = f"Week {week_num}"
            
            emotion_counts = conn.execute(
                """SELECT emotion, COUNT(*) as cnt
                   FROM visit_logs
                   WHERE strftime('%W', checked_in_at) = ?
                   AND strftime('%Y', checked_in_at) = ?
                   GROUP BY emotion ORDER BY emotion""",
                (str(week_num).zfill(2), str(week_start.year)),
            ).fetchall()
            result[week_key] = {r["emotion"]: r["cnt"] for r in emotion_counts}
    
    # Return in reverse chronological order (current week first)
    return {k: result[k] for k in sorted(result.keys(), reverse=True)}


def get_emotion_statistics(days: int = 30) -> dict:
    """Get overall emotion statistics for the past N days.
    
    Returns:
        Dict with emotion labels, counts, and percentages.
    """
    from datetime import timedelta
    
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    
    with get_db() as conn:
        emotion_counts = conn.execute(
            """SELECT emotion, COUNT(*) as cnt
               FROM visit_logs
               WHERE DATE(checked_in_at) >= ?
               GROUP BY emotion ORDER BY cnt DESC""",
            (start_date,),
        ).fetchall()
        
        total = sum(r["cnt"] for r in emotion_counts)
    
    result = {}
    for row in emotion_counts:
        emotion = row["emotion"]
        count = row["cnt"]
        percentage = (count / total * 100) if total > 0 else 0
        result[emotion] = {
            "count": count,
            "percentage": round(percentage, 1)
        }
    
    return result

