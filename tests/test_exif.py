from datetime import datetime
from PIL import Image
import piexif
from photosearch.exif import extract_exif


def _img_with_exif(tmp_path):
    exif = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}
    exif["Exif"][piexif.ExifIFD.DateTimeOriginal] = b"2024:07:14 10:30:00"
    exif["0th"][piexif.ImageIFD.Model] = b"TestCam"
    exif["0th"][piexif.ImageIFD.Orientation] = 1
    p = tmp_path / "e.jpg"
    Image.new("RGB", (10, 10), "blue").save(p, exif=piexif.dump(exif))
    return Image.open(p)


def test_extract_reads_datetime_and_camera(tmp_path):
    data = extract_exif(_img_with_exif(tmp_path))
    assert data.taken_at == datetime(2024, 7, 14, 10, 30, 0)
    assert data.camera == "TestCam"
    assert data.orientation == 1


def test_extract_no_exif_returns_all_none():
    data = extract_exif(Image.new("RGB", (10, 10)))
    assert data.taken_at is None and data.gps_lat is None and data.camera is None
