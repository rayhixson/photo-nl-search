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


def test_passes_filter():
    from photosearch.faces import passes_filter
    assert passes_filter(0.9, (0, 0, 100, 100), 0.6, 50) is True   # confident + big
    assert passes_filter(0.4, (0, 0, 100, 100), 0.6, 50) is False  # low confidence
    assert passes_filter(0.9, (0, 0, 30, 30), 0.6, 50) is False    # too small
    assert passes_filter(0.6, (0, 0, 50, 50), 0.6, 50) is True     # exactly at thresholds
