from pathlib import Path

from PIL import Image, ImageOps
import pillow_heif

_registered = False


def register_heif() -> None:
    global _registered
    if not _registered:
        pillow_heif.register_heif_opener()
        _registered = True


def make_thumbnail(img: Image.Image, photo_id: str, thumb_dir: Path, max_size: int = 512) -> Path:
    thumb_dir = Path(thumb_dir)
    thumb_dir.mkdir(parents=True, exist_ok=True)
    im = ImageOps.exif_transpose(img).convert("RGB")
    im.thumbnail((max_size, max_size))
    out = thumb_dir / f"{photo_id}.jpg"
    im.save(out, "JPEG", quality=85)
    return out
