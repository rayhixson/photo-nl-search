# tests/test_db.py
from photosearch import db as db_module


def test_connect_loads_sqlite_vec_and_schema(db):
    # sqlite-vec is loaded: vec_version() exists
    (version,) = db.execute("SELECT vec_version()").fetchone()
    assert isinstance(version, str)
    # all expected tables and virtual tables exist
    names = {r[0] for r in db.execute(
        "SELECT name FROM sqlite_master WHERE type IN ('table','view')"
    )}
    assert {"photos", "faces", "people", "meta", "photo_vectors", "face_vectors"} <= names


def test_init_schema_is_idempotent(db):
    db_module.init_schema(db)  # must not raise (fixture already called it once)


def test_photo_vectors_roundtrip(db):
    import numpy as np
    db.execute(
        "INSERT INTO photos(id, path, content_hash) VALUES (?,?,?)",
        ("p1", "/a.jpg", "h1"),
    )
    vec = np.ones(db_module.PHOTO_DIM, dtype="float32")
    db.execute(
        "INSERT INTO photo_vectors(photo_id, embedding) VALUES (?, ?)",
        ("p1", vec.tobytes()),
    )
    db.commit()
    (count,) = db.execute("SELECT count(*) FROM photo_vectors").fetchone()
    assert count == 1
