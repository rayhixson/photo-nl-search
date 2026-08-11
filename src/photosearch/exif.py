from dataclasses import dataclass
from datetime import datetime

from PIL import Image, ExifTags

_GPS = {v: k for k, v in ExifTags.GPSTAGS.items()}


@dataclass
class ExifData:
    taken_at: datetime | None = None
    gps_lat: float | None = None
    gps_lng: float | None = None
    camera: str | None = None
    orientation: int | None = None


def _to_deg(dms, ref) -> float | None:
    try:
        d, m, s = (float(x) for x in dms)
        val = d + m / 60 + s / 3600
        return -val if ref in ("S", "W") else val
    except Exception:
        return None


def extract_exif(img: Image.Image) -> ExifData:
    out = ExifData()
    try:
        exif = img.getexif()
    except Exception:
        return out
    if not exif:
        return out
    tags = {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}
    out.camera = tags.get("Model").strip() if isinstance(tags.get("Model"), str) else (
        tags.get("Model").decode(errors="ignore").strip() if isinstance(tags.get("Model"), bytes) else None
    )
    out.orientation = tags.get("Orientation") if isinstance(tags.get("Orientation"), int) else None

    ifd = exif.get_ifd(ExifTags.IFD.Exif) if hasattr(ExifTags, "IFD") else {}
    dt_raw = ifd.get(ExifTags.Base.DateTimeOriginal) if hasattr(ExifTags, "Base") else None
    if isinstance(dt_raw, bytes):
        dt_raw = dt_raw.decode(errors="ignore")
    if isinstance(dt_raw, str):
        try:
            out.taken_at = datetime.strptime(dt_raw.strip(), "%Y:%m:%d %H:%M:%S")
        except ValueError:
            out.taken_at = None

    try:
        gps = exif.get_ifd(ExifTags.IFD.GPSInfo)
    except Exception:
        gps = None
    if gps:
        lat = _to_deg(gps.get(_GPS["GPSLatitude"]), gps.get(_GPS["GPSLatitudeRef"]))
        lng = _to_deg(gps.get(_GPS["GPSLongitude"]), gps.get(_GPS["GPSLongitudeRef"]))
        out.gps_lat, out.gps_lng = lat, lng
    return out
