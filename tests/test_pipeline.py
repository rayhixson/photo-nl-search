# tests/test_pipeline.py
from PIL import Image
from photosearch.pipeline import ingest_photo


def _photo(tmp_path, name="p.jpg", color="red"):
    p = tmp_path / name
    Image.new("RGB", (50, 50), color).save(p)
    return p


def test_ingest_writes_all_tables(db, tmp_path, fake_embedder, fake_detector):
    p = _photo(tmp_path)
    pid = ingest_photo(p, db, fake_embedder, fake_detector, tmp_path / "thumbs")
    assert pid is not None
    assert db.execute("SELECT count(*) FROM photos").fetchone()[0] == 1
    assert db.execute("SELECT count(*) FROM photo_vectors").fetchone()[0] == 1
    assert db.execute("SELECT count(*) FROM faces").fetchone()[0] == 1
    assert db.execute("SELECT count(*) FROM face_vectors").fetchone()[0] == 1
    row = db.execute("SELECT thumb_path FROM photos").fetchone()
    from pathlib import Path
    assert Path(row["thumb_path"]).exists()


def test_ingest_is_idempotent(db, tmp_path, fake_embedder, fake_detector):
    p = _photo(tmp_path)
    ingest_photo(p, db, fake_embedder, fake_detector, tmp_path / "thumbs")
    second = ingest_photo(p, db, fake_embedder, fake_detector, tmp_path / "thumbs")
    assert second is None
    assert db.execute("SELECT count(*) FROM photos").fetchone()[0] == 1
