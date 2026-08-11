# tests/test_db.py
from photosearch import db


def test_connect_loads_sqlite_vec_and_schema(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    db.init_schema(conn)
    # sqlite-vec is loaded: vec_version() exists
    (version,) = conn.execute("SELECT vec_version()").fetchone()
    assert isinstance(version, str)
    # all expected tables exist
    names = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type IN ('table','view')"
    )}
    assert {"photos", "faces", "people", "meta"} <= names


def test_init_schema_is_idempotent(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    db.init_schema(conn)
    db.init_schema(conn)  # must not raise


def test_photo_vectors_roundtrip(tmp_path):
    import numpy as np
    conn = db.connect(tmp_path / "t.db")
    db.init_schema(conn)
    conn.execute(
        "INSERT INTO photos(id, path, content_hash) VALUES (?,?,?)",
        ("p1", "/a.jpg", "h1"),
    )
    vec = np.ones(db.PHOTO_DIM, dtype="float32")
    conn.execute(
        "INSERT INTO photo_vectors(photo_id, embedding) VALUES (?, ?)",
        ("p1", vec.tobytes()),
    )
    conn.commit()
    (count,) = conn.execute("SELECT count(*) FROM photo_vectors").fetchone()
    assert count == 1
