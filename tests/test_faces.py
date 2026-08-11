import numpy as np
import pytest
from PIL import Image
from photosearch import db


@pytest.mark.integration
def test_insightface_returns_faces_with_embeddings():
    from photosearch.faces import InsightFaceDetector
    det = InsightFaceDetector()
    faces = det.detect(Image.new("RGB", (640, 640)))  # no faces -> empty
    assert isinstance(faces, list)


def test_fake_detector_shape(fake_detector):
    faces = fake_detector.detect(Image.new("RGB", (100, 100)))
    assert len(faces) == 1
    f = faces[0]
    assert f.embedding.shape == (db.FACE_DIM,)
    assert abs(np.linalg.norm(f.embedding) - 1.0) < 1e-5
    assert len(f.bbox) == 4
