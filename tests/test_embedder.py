import numpy as np
import pytest
from PIL import Image
from photosearch import db


@pytest.mark.integration
def test_clip_embedder_shapes_and_norm():
    from photosearch.embedder import ClipEmbedder
    emb = ClipEmbedder("MobileCLIP-S1", "datacompdr", device="mps")
    v = emb.embed_image(Image.new("RGB", (64, 64), "red"))
    t = emb.embed_text("a red square")
    assert v.shape == (db.PHOTO_DIM,) and v.dtype == np.float32
    assert t.shape == (db.PHOTO_DIM,)
    assert abs(np.linalg.norm(v) - 1.0) < 1e-3


def test_fake_embedder_is_deterministic_and_normalized(fake_embedder):
    a = fake_embedder.embed_text("dog")
    b = fake_embedder.embed_text("dog")
    assert np.allclose(a, b)
    assert abs(np.linalg.norm(a) - 1.0) < 1e-5
    assert a.shape == (db.PHOTO_DIM,)
