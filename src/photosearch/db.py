# src/photosearch/db.py
import sqlite3
from pathlib import Path

import sqlite_vec

PHOTO_DIM = 512
FACE_DIM = 512


def connect(db_path: Path) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        f"""
        CREATE TABLE IF NOT EXISTS photos (
            id TEXT PRIMARY KEY,
            path TEXT UNIQUE,
            content_hash TEXT UNIQUE,
            taken_at TEXT,
            gps_lat REAL, gps_lng REAL,
            camera TEXT,
            width INTEGER, height INTEGER,
            thumb_path TEXT,
            mtime REAL, size INTEGER
        );
        CREATE VIRTUAL TABLE IF NOT EXISTS photo_vectors USING vec0(
            photo_id TEXT PRIMARY KEY,
            embedding float[{PHOTO_DIM}]
        );
        CREATE TABLE IF NOT EXISTS people (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT
        );
        CREATE TABLE IF NOT EXISTS faces (
            id TEXT PRIMARY KEY,
            photo_id TEXT REFERENCES photos(id) ON DELETE CASCADE,
            bbox TEXT,
            person_id INTEGER REFERENCES people(id)
        );
        CREATE VIRTUAL TABLE IF NOT EXISTS face_vectors USING vec0(
            face_id TEXT PRIMARY KEY,
            embedding float[{FACE_DIM}]
        );
        CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
        """
    )
    conn.commit()
