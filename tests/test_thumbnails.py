from PIL import Image
from photosearch.thumbnails import make_thumbnail


def test_make_thumbnail_writes_bounded_jpeg(tmp_path):
    img = Image.new("RGB", (2000, 1000), "green")
    out = make_thumbnail(img, "abc123", tmp_path, max_size=256)
    assert out.exists() and out.suffix == ".jpg"
    with Image.open(out) as t:
        assert max(t.size) <= 256
        assert t.format == "JPEG"
