import hashlib

import numpy as np
import pytest
from photosearch import db as _db
from photosearch.db import PHOTO_DIM


@pytest.fixture
def db(tmp_path):
    conn = _db.connect(tmp_path / "test.db")
    _db.init_schema(conn)
    yield conn
    conn.close()


class FakeEmbedder:
    def _vec(self, seed: str) -> np.ndarray:
        h = hashlib.sha256(seed.encode()).digest()
        rng = np.random.default_rng(int.from_bytes(h[:8], "little"))
        v = rng.standard_normal(PHOTO_DIM).astype("float32")
        return v / np.linalg.norm(v)

    def embed_image(self, img):
        return self._vec(f"img-{img.size}-{img.getpixel((0, 0))}")

    def embed_text(self, text):
        return self._vec(f"text-{text}")


@pytest.fixture
def fake_embedder():
    return FakeEmbedder()


from photosearch.faces import Face
from photosearch.db import FACE_DIM


class FakeFaceDetector:
    def __init__(self, per_image=1):
        self.per_image = per_image

    def detect(self, img):
        faces = []
        for i in range(self.per_image):
            rng = np.random.default_rng(1000 + i)
            v = rng.standard_normal(FACE_DIM).astype("float32")
            faces.append(Face(bbox=(i, i, i + 10, i + 10), embedding=v / np.linalg.norm(v)))
        return faces


@pytest.fixture
def fake_detector():
    return FakeFaceDetector()
