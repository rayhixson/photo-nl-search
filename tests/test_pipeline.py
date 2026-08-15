# tests/test_pipeline.py
from pathlib import Path

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
    assert Path(row["thumb_path"]).exists()


def test_ingest_is_idempotent(db, tmp_path, fake_embedder, fake_detector):
    p = _photo(tmp_path)
    ingest_photo(p, db, fake_embedder, fake_detector, tmp_path / "thumbs")
    second = ingest_photo(p, db, fake_embedder, fake_detector, tmp_path / "thumbs")
    assert second is None
    assert db.execute("SELECT count(*) FROM photos").fetchone()[0] == 1


def test_reingest_modified_file_no_orphans(db, tmp_path, fake_embedder, fake_detector):
    """Re-ingesting a modified file (same path, new hash) must leave no orphaned vec rows."""
    thumb_dir = tmp_path / "thumbs"
    p = _photo(tmp_path, color="red")
    pid1 = ingest_photo(p, db, fake_embedder, fake_detector, thumb_dir)
    assert pid1 is not None

    # Overwrite the file with different content so the hash changes
    Image.new("RGB", (50, 50), "blue").save(p)
    pid2 = ingest_photo(p, db, fake_embedder, fake_detector, thumb_dir)
    assert pid2 is not None
    assert pid2 != pid1

    # No orphans — each vec table should contain exactly the rows for the new content
    assert db.execute("SELECT count(*) FROM photos").fetchone()[0] == 1
    assert db.execute("SELECT count(*) FROM photo_vectors").fetchone()[0] == 1
    face_count = db.execute("SELECT count(*) FROM faces").fetchone()[0]
    assert db.execute("SELECT count(*) FROM face_vectors").fetchone()[0] == face_count
    # All remaining faces belong to the new photo
    orphaned_faces = db.execute(
        "SELECT count(*) FROM faces WHERE photo_id != ?", (pid2,)
    ).fetchone()[0]
    assert orphaned_faces == 0
