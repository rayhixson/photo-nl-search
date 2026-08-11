# src/photosearch/pipeline.py
import hashlib
import json
from pathlib import Path

from PIL import Image

from .embedder import Embedder
from .exif import extract_exif
from .faces import FaceDetector
from .thumbnails import make_thumbnail, register_heif


def _hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ingest_photo(path: Path, conn, embedder: Embedder, detector: FaceDetector, thumb_dir: Path) -> str | None:
    register_heif()
    path = Path(path)
    try:
        content_hash = _hash(path)
    except OSError:
        return None
    if conn.execute("SELECT 1 FROM photos WHERE content_hash = ?", (content_hash,)).fetchone():
        return None

    try:
        with Image.open(path) as img:
            img.load()
            exif = extract_exif(img)
            width, height = img.size
            thumb = make_thumbnail(img, content_hash, thumb_dir)
            photo_vec = embedder.embed_image(img)
            faces = detector.detect(img)
    except Exception:
        return None

    st = path.stat()
    pid = content_hash
    conn.execute(
        """INSERT OR REPLACE INTO photos
           (id, path, content_hash, taken_at, gps_lat, gps_lng, camera, width, height, thumb_path, mtime, size)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
        (pid, str(path), content_hash,
         exif.taken_at.isoformat() if exif.taken_at else None,
         exif.gps_lat, exif.gps_lng, exif.camera, width, height, str(thumb),
         st.st_mtime, st.st_size),
    )
    conn.execute("DELETE FROM photo_vectors WHERE photo_id = ?", (pid,))
    conn.execute("INSERT INTO photo_vectors(photo_id, embedding) VALUES (?, ?)",
                 (pid, photo_vec.tobytes()))
    for i, face in enumerate(faces):
        fid = f"{pid}:{i}"
        conn.execute("INSERT INTO faces(id, photo_id, bbox, person_id) VALUES (?,?,?,NULL)",
                     (fid, pid, json.dumps(face.bbox)))
        conn.execute("INSERT INTO face_vectors(face_id, embedding) VALUES (?, ?)",
                     (fid, face.embedding.tobytes()))
    conn.commit()
    return pid
