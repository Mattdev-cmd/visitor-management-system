"""
SQLite database helpers — schema creation & convenience wrappers.
"""

import sqlite3
import os
import hashlib
from contextlib import contextmanager

import config


def get_connection() -> sqlite3.Connection:
    """Return a new connection with row-factory enabled."""
    conn = sqlite3.connect(config.DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")  # better concurrent reads
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def get_db():
    """Context-managed database connection."""
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """Create tables if they don't exist."""
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS visitors (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                name            TEXT    NOT NULL,
                purpose         TEXT    NOT NULL,
                contact         TEXT,
                photo_path      TEXT,
                face_encoding   BLOB,
                created_at      TEXT    DEFAULT (datetime('now','localtime'))
            );

            CREATE TABLE IF NOT EXISTS visit_logs (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                visitor_id      INTEGER NOT NULL,
                emotion         TEXT,
                confidence      REAL,
                photo_path      TEXT,
                checked_in_at   TEXT    DEFAULT (datetime('now','localtime')),
                checked_out_at  TEXT,
                FOREIGN KEY (visitor_id) REFERENCES visitors(id)
            );

            CREATE INDEX IF NOT EXISTS idx_visit_logs_visitor
                ON visit_logs(visitor_id);
            CREATE INDEX IF NOT EXISTS idx_visit_logs_checkin
                ON visit_logs(checked_in_at);

            CREATE TABLE IF NOT EXISTS users (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                username        TEXT    NOT NULL UNIQUE,
                password_hash   TEXT    NOT NULL,
                role            TEXT    NOT NULL DEFAULT 'admin',
                created_at      TEXT    DEFAULT (datetime('now','localtime'))
            );
            CREATE TABLE IF NOT EXISTS pre_registrations (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                name            TEXT    NOT NULL,
                purpose         TEXT    NOT NULL,
                contact         TEXT,
                photo_path      TEXT,
                status          TEXT    NOT NULL DEFAULT 'pending',
                submitted_at    TEXT    DEFAULT (datetime('now','localtime')),
                reviewed_at     TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_prereg_status
                ON pre_registrations(status);        """)
        # Create default superadmin account if no users exist
        count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        if count == 0:
            pw_hash = hashlib.sha256("admin123".encode()).hexdigest()
            conn.execute("INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                         ("admin", pw_hash, "superadmin"))
    print("[DB] Database initialised at", config.DATABASE_PATH)
