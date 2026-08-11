# Photo Natural-Language Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fully-local Mac app that indexes NAS photos and answers natural-language queries (including people) through a local web UI.

**Architecture:** A Python background service scans SMB-mounted photo folders, ingests each photo (EXIF + thumbnail + MobileCLIP embedding + face embeddings) into a single SQLite database with the `sqlite-vec` extension, clusters faces into named people, and serves a FastAPI + browser UI. Queries are parsed by a local Ollama model into structured filters, then answered with a `sqlite-vec` KNN search joined to metadata.

**Tech Stack:** Python 3.11+, FastAPI + uvicorn, SQLite + `sqlite-vec`, `open_clip` (MobileCLIP) on PyTorch MPS, InsightFace on ONNX Runtime, scikit-learn HDBSCAN, Ollama (Qwen2.5 3B), Pillow + pillow-heif.

## Global Constraints

- macOS Apple Silicon; all inference runs locally (no cloud calls). No Docker — run natively to keep Metal/MPS acceleration.
- Environment is a `uv`-managed venv pinned to **Python 3.12** (`uv venv --python 3.12`). The host default Python 3.14 lacks ML-stack wheels; do not use it.
- **All Python/test commands run inside the venv** — prefix with `uv run` (e.g. `uv run pytest`) or activate `.venv` first. Wherever a step says `pytest` or `python`, it means `uv run pytest` / `uv run python`.
- Single SQLite file for all data; vectors via the `sqlite-vec` loadable extension.
- SQLite extension loading must be enabled — the `uv`-managed CPython 3.12 allows it by default; if a step hits `AttributeError: enable_load_extension`, fall back to `pip install pysqlite3-binary`.
- Photo embedding dimension `PHOTO_DIM = 512`; face embedding dimension `FACE_DIM = 512`. All stored embeddings are float32 and L2-normalized.
- Media scope: JPEG / PNG / HEIC only.
- NAS is read-only; never write to share paths.
- **No git for this project (user preference).** Every task ends with a "Checkpoint" step that runs the full test suite instead of a commit.
- Package name: `photosearch`, importable from `src/` (installed with `pip install -e .`).
- Test runner: `pytest`. Heavy models (MobileCLIP, InsightFace, Ollama) are faked in unit tests via the interfaces defined below; real-model tests are marked `@pytest.mark.integration`.

## File Structure

```
photo-nl-search/
├── pyproject.toml                 # deps + package config + pytest markers
├── config.example.toml            # sample user config
├── src/photosearch/
│   ├── __init__.py
│   ├── config.py                  # Config dataclass + load_config()
│   ├── db.py                      # connect(), init_schema(), dim constants
│   ├── exif.py                    # extract_exif() -> ExifData
│   ├── thumbnails.py              # make_thumbnail()
│   ├── embedder.py                # Embedder: embed_image/embed_text (Protocol + impl)
│   ├── faces.py                   # FaceDetector: detect() (Protocol + impl)
│   ├── scanner.py                 # scan_for_changes()
│   ├── pipeline.py                # ingest_photo()
│   ├── clustering.py              # cluster_faces()
│   ├── query/
│   │   ├── __init__.py
│   │   ├── parser.py              # parse_query() -> QueryFilters (Ollama)
│   │   └── search.py              # search() -> list[SearchResult]
│   ├── api.py                     # FastAPI app + endpoints, serves web/
│   └── service.py                 # background scan loop + uvicorn entrypoint
├── web/
│   ├── index.html
│   ├── app.js
│   └── style.css
├── tests/
│   ├── conftest.py                # fixtures: temp db, fake embedder/detector, sample images
│   ├── test_config.py
│   ├── test_db.py
│   ├── test_exif.py
│   ├── test_thumbnails.py
│   ├── test_embedder.py
│   ├── test_faces.py
│   ├── test_scanner.py
│   ├── test_pipeline.py
│   ├── test_clustering.py
│   ├── test_parser.py
│   ├── test_search.py
│   └── test_api.py
└── docs/                          # spec + this plan (already present)
```

---

### Task 1: Project scaffolding

**Files:**
- Create: `pyproject.toml`, `src/photosearch/__init__.py`, `src/photosearch/query/__init__.py`, `tests/conftest.py`, `config.example.toml`

**Interfaces:**
- Consumes: nothing.
- Produces: an installable `photosearch` package and a `pytest` setup with markers.

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "photosearch"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.110",
    "uvicorn[standard]>=0.29",
    "pillow>=10.2",
    "pillow-heif>=0.15",
    "open_clip_torch>=2.24",
    "torch>=2.2",
    "onnxruntime>=1.17",
    "insightface>=0.7.3",
    "scikit-learn>=1.4",
    "numpy>=1.26",
    "sqlite-vec>=0.1.1",
    "httpx>=0.27",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-asyncio>=0.23", "piexif>=1.1.3"]

[project.scripts]
photosearch = "photosearch.service:main"

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
markers = ["integration: tests that load real models or hit Ollama (slow)"]
asyncio_mode = "auto"
addopts = "-m 'not integration'"
```

- [ ] **Step 2: Create empty `src/photosearch/__init__.py` and `src/photosearch/query/__init__.py`**

Both files are empty.

- [ ] **Step 3: Write `config.example.toml`**

```toml
share_paths = ["/Volumes/nas-photos"]
db_path = "~/.photosearch/photos.db"
thumb_dir = "~/.photosearch/thumbs"
clip_model = "MobileCLIP-S1"
clip_pretrained = "datacompdr"
ollama_url = "http://localhost:11434"
ollama_model = "qwen2.5:3b"
scan_interval_s = 900
```

- [ ] **Step 4: Create the venv and install**

Run: `uv venv --python 3.12 && uv pip install -e '.[dev]'`
Expected: venv created at `.venv`, install succeeds. (`.venv` is not committed — add it to `.gitignore` if not already ignored.)

- [ ] **Step 5: Verify the package imports**

Run: `uv run python -c "import photosearch"`
Expected: no error.

- [ ] **Step 6: Checkpoint**

Run: `uv run pytest` (0 tests collected is fine at this point; command must exit 0).

---

### Task 2: Config loading

**Files:**
- Create: `src/photosearch/config.py`, `tests/test_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `@dataclass Config` with fields: `share_paths: list[Path]`, `db_path: Path`, `thumb_dir: Path`, `clip_model: str`, `clip_pretrained: str`, `ollama_url: str`, `ollama_model: str`, `scan_interval_s: int`.
  - `def load_config(path: Path) -> Config` — reads TOML, expands `~`, resolves paths.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
from pathlib import Path
from photosearch.config import load_config

def test_load_config_expands_and_parses(tmp_path):
    cfg_file = tmp_path / "c.toml"
    cfg_file.write_text(
        'share_paths = ["/Volumes/nas"]\n'
        'db_path = "~/.photosearch/photos.db"\n'
        'thumb_dir = "~/.photosearch/thumbs"\n'
        'clip_model = "MobileCLIP-S1"\n'
        'clip_pretrained = "datacompdr"\n'
        'ollama_url = "http://localhost:11434"\n'
        'ollama_model = "qwen2.5:3b"\n'
        'scan_interval_s = 900\n'
    )
    cfg = load_config(cfg_file)
    assert cfg.share_paths == [Path("/Volumes/nas")]
    assert cfg.db_path == Path.home() / ".photosearch" / "photos.db"
    assert cfg.scan_interval_s == 900
    assert cfg.clip_model == "MobileCLIP-S1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: photosearch.config`.

- [ ] **Step 3: Write the implementation**

```python
# src/photosearch/config.py
import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    share_paths: list[Path]
    db_path: Path
    thumb_dir: Path
    clip_model: str
    clip_pretrained: str
    ollama_url: str
    ollama_model: str
    scan_interval_s: int


def _p(value: str) -> Path:
    return Path(value).expanduser()


def load_config(path: Path) -> Config:
    data = tomllib.loads(Path(path).read_text())
    return Config(
        share_paths=[_p(s) for s in data["share_paths"]],
        db_path=_p(data["db_path"]),
        thumb_dir=_p(data["thumb_dir"]),
        clip_model=data["clip_model"],
        clip_pretrained=data["clip_pretrained"],
        ollama_url=data["ollama_url"],
        ollama_model=data["ollama_model"],
        scan_interval_s=int(data["scan_interval_s"]),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py -v`
Expected: PASS.

- [ ] **Step 5: Checkpoint**

Run: `pytest`
Expected: all tests pass.

---

### Task 3: Database layer + schema

**Files:**
- Create: `src/photosearch/db.py`, `tests/test_db.py`
- Modify: `tests/conftest.py` (add `db` fixture)

**Interfaces:**
- Consumes: nothing.
- Produces:
  - Constants `PHOTO_DIM = 512`, `FACE_DIM = 512`.
  - `def connect(db_path: Path) -> sqlite3.Connection` — enables extension loading, loads `sqlite-vec`, sets `row_factory = sqlite3.Row`, enables foreign keys.
  - `def init_schema(conn: sqlite3.Connection) -> None` — creates all tables idempotently.
  - Tables: `photos(id TEXT PK, path TEXT UNIQUE, content_hash TEXT UNIQUE, taken_at TEXT, gps_lat REAL, gps_lng REAL, camera TEXT, width INT, height INT, thumb_path TEXT, mtime REAL, size INT)`; `photo_vectors` (vec0, `photo_id` + `embedding float[512]`); `faces(id TEXT PK, photo_id TEXT REFERENCES photos(id), bbox TEXT, person_id INT)`; `face_vectors` (vec0, `face_id` + `embedding float[512]`); `people(id INTEGER PK, name TEXT)`; `meta(key TEXT PK, value TEXT)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_db.py
from photosearch import db

def test_connect_loads_sqlite_vec_and_schema(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    db.init_schema(conn)
    # sqlite-vec is loaded: vec_version() exists
    (version,) = conn.execute("SELECT vec_version()").fetchone()
    assert isinstance(version, str)
    # all expected tables exist
    names = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type IN ('table','view')"
    )}
    assert {"photos", "faces", "people", "meta"} <= names

def test_init_schema_is_idempotent(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    db.init_schema(conn)
    db.init_schema(conn)  # must not raise

def test_photo_vectors_roundtrip(tmp_path):
    import numpy as np
    conn = db.connect(tmp_path / "t.db")
    db.init_schema(conn)
    conn.execute(
        "INSERT INTO photos(id, path, content_hash) VALUES (?,?,?)",
        ("p1", "/a.jpg", "h1"),
    )
    vec = np.ones(db.PHOTO_DIM, dtype="float32")
    conn.execute(
        "INSERT INTO photo_vectors(photo_id, embedding) VALUES (?, ?)",
        ("p1", vec.tobytes()),
    )
    conn.commit()
    (count,) = conn.execute("SELECT count(*) FROM photo_vectors").fetchone()
    assert count == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_db.py -v`
Expected: FAIL — module/attribute missing.

- [ ] **Step 3: Write the implementation**

```python
# src/photosearch/db.py
import sqlite3
from pathlib import Path

import sqlite_vec

PHOTO_DIM = 512
FACE_DIM = 512


def connect(db_path: Path) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        f"""
        CREATE TABLE IF NOT EXISTS photos (
            id TEXT PRIMARY KEY,
            path TEXT UNIQUE,
            content_hash TEXT UNIQUE,
            taken_at TEXT,
            gps_lat REAL, gps_lng REAL,
            camera TEXT,
            width INTEGER, height INTEGER,
            thumb_path TEXT,
            mtime REAL, size INTEGER
        );
        CREATE VIRTUAL TABLE IF NOT EXISTS photo_vectors USING vec0(
            photo_id TEXT PRIMARY KEY,
            embedding float[{PHOTO_DIM}]
        );
        CREATE TABLE IF NOT EXISTS people (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT
        );
        CREATE TABLE IF NOT EXISTS faces (
            id TEXT PRIMARY KEY,
            photo_id TEXT REFERENCES photos(id) ON DELETE CASCADE,
            bbox TEXT,
            person_id INTEGER REFERENCES people(id)
        );
        CREATE VIRTUAL TABLE IF NOT EXISTS face_vectors USING vec0(
            face_id TEXT PRIMARY KEY,
            embedding float[{FACE_DIM}]
        );
        CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
        """
    )
    conn.commit()
```

- [ ] **Step 4: Add the `db` fixture to conftest**

```python
# tests/conftest.py
import pytest
from photosearch import db as _db

@pytest.fixture
def db(tmp_path):
    conn = _db.connect(tmp_path / "test.db")
    _db.init_schema(conn)
    yield conn
    conn.close()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_db.py -v`
Expected: PASS. If `enable_load_extension` raises `AttributeError`, the Python build blocks extensions — install `pysqlite3-binary` or a Homebrew Python and retry (see Global Constraints).

- [ ] **Step 6: Checkpoint**

Run: `pytest`
Expected: all tests pass.

---

### Task 4: EXIF extraction

**Files:**
- Create: `src/photosearch/exif.py`, `tests/test_exif.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `@dataclass ExifData` fields: `taken_at: datetime | None`, `gps_lat: float | None`, `gps_lng: float | None`, `camera: str | None`, `orientation: int | None`.
  - `def extract_exif(img: PIL.Image.Image) -> ExifData` — never raises on missing/garbage tags; returns `None` for absent fields.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_exif.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_exif.py -v`
Expected: FAIL — module missing. (`piexif` is a transitive test-only helper; add `piexif` to the dev extra in `pyproject.toml`.)

- [ ] **Step 3: Write the implementation**

```python
# src/photosearch/exif.py
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_exif.py -v`
Expected: PASS.

- [ ] **Step 5: Checkpoint**

Run: `pytest`
Expected: all tests pass.

---

### Task 5: Thumbnail generation

**Files:**
- Create: `src/photosearch/thumbnails.py`, `tests/test_thumbnails.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `def make_thumbnail(img: PIL.Image.Image, photo_id: str, thumb_dir: Path, max_size: int = 512) -> Path` — writes a JPEG honoring EXIF orientation, returns the written path. Registers HEIF so HEIC opens elsewhere.
  - `def register_heif() -> None` — idempotent `pillow_heif` opener registration.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_thumbnails.py
from PIL import Image
from photosearch.thumbnails import make_thumbnail

def test_make_thumbnail_writes_bounded_jpeg(tmp_path):
    img = Image.new("RGB", (2000, 1000), "green")
    out = make_thumbnail(img, "abc123", tmp_path, max_size=256)
    assert out.exists() and out.suffix == ".jpg"
    with Image.open(out) as t:
        assert max(t.size) <= 256
        assert t.format == "JPEG"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_thumbnails.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Write the implementation**

```python
# src/photosearch/thumbnails.py
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_thumbnails.py -v`
Expected: PASS.

- [ ] **Step 5: Checkpoint**

Run: `pytest`
Expected: all tests pass.

---

### Task 6: Image + text embedder (MobileCLIP)

**Files:**
- Create: `src/photosearch/embedder.py`, `tests/test_embedder.py`
- Modify: `tests/conftest.py` (add `FakeEmbedder`)

**Interfaces:**
- Consumes: `db.PHOTO_DIM`.
- Produces:
  - `class Embedder(Protocol)` with `embed_image(img: PIL.Image.Image) -> np.ndarray` and `embed_text(text: str) -> np.ndarray`, each returning a float32 L2-normalized vector of length `PHOTO_DIM`.
  - `class ClipEmbedder(Embedder)` — real impl: `__init__(model_name: str, pretrained: str, device: str = "mps")`, loads `open_clip`, exposes `embed_image` / `embed_text`.
  - `FakeEmbedder` (in conftest) — deterministic hash-based vectors so consumers can be tested without models.

- [ ] **Step 1: Write the failing test (real impl marked integration)**

```python
# tests/test_embedder.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_embedder.py -v`
Expected: FAIL — `fake_embedder` fixture and module missing. (Integration test is deselected by default.)

- [ ] **Step 3: Write the real implementation**

```python
# src/photosearch/embedder.py
from typing import Protocol

import numpy as np
import open_clip
import torch
from PIL import Image

from .db import PHOTO_DIM


class Embedder(Protocol):
    def embed_image(self, img: Image.Image) -> np.ndarray: ...
    def embed_text(self, text: str) -> np.ndarray: ...


def _normalize(v: np.ndarray) -> np.ndarray:
    v = v.astype("float32")
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


class ClipEmbedder:
    def __init__(self, model_name: str, pretrained: str, device: str = "mps"):
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model = self.model.to(device).eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)
        assert self.model.visual.output_dim == PHOTO_DIM, "model dim != PHOTO_DIM"

    @torch.no_grad()
    def embed_image(self, img: Image.Image) -> np.ndarray:
        x = self.preprocess(img.convert("RGB")).unsqueeze(0).to(self.device)
        feats = self.model.encode_image(x)[0].cpu().numpy()
        return _normalize(feats)

    @torch.no_grad()
    def embed_text(self, text: str) -> np.ndarray:
        toks = self.tokenizer([text]).to(self.device)
        feats = self.model.encode_text(toks)[0].cpu().numpy()
        return _normalize(feats)
```

- [ ] **Step 4: Add `FakeEmbedder` + fixture to conftest**

```python
# tests/conftest.py  (append)
import hashlib
import numpy as np
from photosearch.db import PHOTO_DIM

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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_embedder.py -v`
Expected: PASS (integration test deselected). Optionally run `pytest -m integration tests/test_embedder.py` once models are available.

- [ ] **Step 6: Checkpoint**

Run: `pytest`
Expected: all tests pass.

---

### Task 7: Face detection + embedding (InsightFace)

**Files:**
- Create: `src/photosearch/faces.py`, `tests/test_faces.py`
- Modify: `tests/conftest.py` (add `FakeFaceDetector`)

**Interfaces:**
- Consumes: `db.FACE_DIM`.
- Produces:
  - `@dataclass Face` fields: `bbox: tuple[int, int, int, int]`, `embedding: np.ndarray` (float32, L2-normalized, length `FACE_DIM`).
  - `class FaceDetector(Protocol)` with `detect(img: PIL.Image.Image) -> list[Face]`.
  - `class InsightFaceDetector(FaceDetector)` — real impl wrapping `insightface.app.FaceAnalysis`.
  - `FakeFaceDetector` (conftest) — returns a scripted list of faces per image for deterministic tests.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_faces.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_faces.py -v`
Expected: FAIL — module + fixture missing.

- [ ] **Step 3: Write the real implementation**

```python
# src/photosearch/faces.py
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from PIL import Image

from .db import FACE_DIM


@dataclass
class Face:
    bbox: tuple[int, int, int, int]
    embedding: np.ndarray


class FaceDetector(Protocol):
    def detect(self, img: Image.Image) -> list[Face]: ...


class InsightFaceDetector:
    def __init__(self, model_name: str = "buffalo_l"):
        from insightface.app import FaceAnalysis

        self.app = FaceAnalysis(name=model_name)
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    def detect(self, img: Image.Image) -> list[Face]:
        rgb = np.asarray(img.convert("RGB"))
        bgr = rgb[:, :, ::-1]
        out: list[Face] = []
        for f in self.app.get(bgr):
            emb = f.normed_embedding.astype("float32")
            x1, y1, x2, y2 = (int(v) for v in f.bbox)
            out.append(Face(bbox=(x1, y1, x2, y2), embedding=emb))
        return out
```

- [ ] **Step 4: Add `FakeFaceDetector` + fixture to conftest**

```python
# tests/conftest.py  (append)
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_faces.py -v`
Expected: PASS (integration deselected).

- [ ] **Step 6: Checkpoint**

Run: `pytest`
Expected: all tests pass.

---

### Task 8: Incremental scanner

**Files:**
- Create: `src/photosearch/scanner.py`, `tests/test_scanner.py`

**Interfaces:**
- Consumes: a `sqlite3.Connection` with the schema from Task 3 (reads `photos.path/mtime/size`).
- Produces:
  - `SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".heic"}`.
  - `def scan_for_changes(conn, share_paths: list[Path]) -> list[Path]` — walks the shares, returns files that are new or whose `(mtime, size)` differs from the stored row. Case-insensitive extension match. Skips unreadable dirs without raising.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_scanner.py
from photosearch.scanner import scan_for_changes

def _touch(p, content=b"x"):
    p.write_bytes(content)

def test_scan_finds_new_and_changed_skips_unchanged(db, tmp_path):
    share = tmp_path / "share"; share.mkdir()
    a = share / "a.jpg"; _touch(a)
    b = share / "b.png"; _touch(b)
    note = share / "notes.txt"; _touch(note)  # ignored ext

    first = scan_for_changes(db, [share])
    assert set(first) == {a, b}

    # record a as ingested
    st = a.stat()
    db.execute(
        "INSERT INTO photos(id, path, content_hash, mtime, size) VALUES (?,?,?,?,?)",
        ("id-a", str(a), "h", st.st_mtime, st.st_size),
    )
    db.commit()

    second = scan_for_changes(db, [share])
    assert a not in second and b in second

    # modify a -> reappears
    _touch(a, b"xxxx")
    third = scan_for_changes(db, [share])
    assert a in third
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_scanner.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Write the implementation**

```python
# src/photosearch/scanner.py
import os
from pathlib import Path

SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".heic"}


def scan_for_changes(conn, share_paths: list[Path]) -> list[Path]:
    known = {
        row["path"]: (row["mtime"], row["size"])
        for row in conn.execute("SELECT path, mtime, size FROM photos")
    }
    changed: list[Path] = []
    for root in share_paths:
        for dirpath, _dirs, files in os.walk(root, onerror=lambda e: None):
            for name in files:
                if Path(name).suffix.lower() not in SUPPORTED_EXT:
                    continue
                p = Path(dirpath) / name
                try:
                    st = p.stat()
                except OSError:
                    continue
                prev = known.get(str(p))
                if prev is None or prev[0] != st.st_mtime or prev[1] != st.st_size:
                    changed.append(p)
    return changed
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_scanner.py -v`
Expected: PASS.

- [ ] **Step 5: Checkpoint**

Run: `pytest`
Expected: all tests pass.

---

### Task 9: Ingest pipeline

**Files:**
- Create: `src/photosearch/pipeline.py`, `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `db` connection; `Embedder`; `FaceDetector`; `extract_exif`; `make_thumbnail`; `register_heif`.
- Produces:
  - `def ingest_photo(path: Path, conn, embedder: Embedder, detector: FaceDetector, thumb_dir: Path) -> str | None` — idempotent by content hash; returns the `photo_id` (sha256 hex of file bytes) or `None` if the photo was already present or unreadable. Writes rows to `photos`, `photo_vectors`, `faces`, `face_vectors`. Uses `INSERT OR REPLACE` on `photos` keyed by `path` so re-ingest after modification updates in place.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pipeline.py
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
    from pathlib import Path
    assert Path(row["thumb_path"]).exists()

def test_ingest_is_idempotent(db, tmp_path, fake_embedder, fake_detector):
    p = _photo(tmp_path)
    ingest_photo(p, db, fake_embedder, fake_detector, tmp_path / "thumbs")
    second = ingest_photo(p, db, fake_embedder, fake_detector, tmp_path / "thumbs")
    assert second is None
    assert db.execute("SELECT count(*) FROM photos").fetchone()[0] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pipeline.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Write the implementation**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_pipeline.py -v`
Expected: PASS.

- [ ] **Step 5: Checkpoint**

Run: `pytest`
Expected: all tests pass.

---

### Task 10: Face clustering

**Files:**
- Create: `src/photosearch/clustering.py`, `tests/test_clustering.py`

**Interfaces:**
- Consumes: `db` connection; `db.FACE_DIM`.
- Produces:
  - `def cluster_faces(conn, min_cluster_size: int = 3) -> int` — loads all face embeddings, runs `sklearn.cluster.HDBSCAN` (cosine via normalized vectors + euclidean), assigns each cluster a `people` row (unnamed) and sets `faces.person_id`; noise points (label `-1`) get `person_id = NULL`. Returns the number of clusters found. Idempotent: clears prior auto-created unnamed people and reassigns each run.

**v1 limitation (documented):** re-clustering can renumber people; names attached to a person survive only if you don't re-cluster. Acceptable for v1 per the spec.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_clustering.py
import numpy as np
from photosearch.clustering import cluster_faces
from photosearch.db import FACE_DIM

def _add_face(db, fid, vec):
    db.execute("INSERT INTO photos(id, path, content_hash) VALUES (?,?,?)",
               (f"ph-{fid}", f"/{fid}.jpg", f"h-{fid}"))
    db.execute("INSERT INTO faces(id, photo_id, bbox, person_id) VALUES (?,?,?,NULL)",
               (fid, f"ph-{fid}", "[0,0,1,1]"))
    db.execute("INSERT INTO face_vectors(face_id, embedding) VALUES (?,?)",
               (fid, vec.astype("float32").tobytes()))

def test_cluster_groups_similar_faces(db):
    base_a = np.zeros(FACE_DIM, dtype="float32"); base_a[0] = 1.0
    base_b = np.zeros(FACE_DIM, dtype="float32"); base_b[1] = 1.0
    rng = np.random.default_rng(0)
    for i in range(4):
        _add_face(db, f"a{i}", base_a + rng.normal(0, 0.01, FACE_DIM))
    for i in range(4):
        _add_face(db, f"b{i}", base_b + rng.normal(0, 0.01, FACE_DIM))
    db.commit()

    n = cluster_faces(db, min_cluster_size=3)
    assert n == 2
    a_ids = {r["person_id"] for r in db.execute("SELECT person_id FROM faces WHERE id LIKE 'a%'")}
    assert len(a_ids) == 1 and None not in a_ids
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_clustering.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Write the implementation**

```python
# src/photosearch/clustering.py
import numpy as np
from sklearn.cluster import HDBSCAN

from .db import FACE_DIM


def cluster_faces(conn, min_cluster_size: int = 3) -> int:
    rows = list(conn.execute("SELECT face_id, embedding FROM face_vectors"))
    if not rows:
        return 0
    face_ids = [r["face_id"] for r in rows]
    vecs = np.stack([np.frombuffer(r["embedding"], dtype="float32") for r in rows])

    labels = HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean").fit_predict(vecs)

    # reset prior auto assignments + unnamed people
    conn.execute("UPDATE faces SET person_id = NULL")
    conn.execute("DELETE FROM people WHERE name IS NULL")

    cluster_to_person: dict[int, int] = {}
    for fid, label in zip(face_ids, labels):
        label = int(label)
        if label < 0:
            continue
        if label not in cluster_to_person:
            cur = conn.execute("INSERT INTO people(name) VALUES (NULL)")
            cluster_to_person[label] = cur.lastrowid
        conn.execute("UPDATE faces SET person_id = ? WHERE id = ?",
                     (cluster_to_person[label], fid))
    conn.commit()
    return len(cluster_to_person)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_clustering.py -v`
Expected: PASS.

- [ ] **Step 5: Checkpoint**

Run: `pytest`
Expected: all tests pass.

---

### Task 11: Query parser (Ollama)

**Files:**
- Create: `src/photosearch/query/parser.py`, `tests/test_parser.py`

**Interfaces:**
- Consumes: `httpx`; Ollama HTTP API at `{ollama_url}/api/chat` with `format="json"`.
- Produces:
  - `@dataclass QueryFilters` fields: `semantic_text: str`, `date_from: date | None`, `date_to: date | None`, `location: str | None`, `people: list[str]`.
  - `async def parse_query(text: str, ollama_url: str, model: str, client: httpx.AsyncClient | None = None) -> QueryFilters` — prompts the model for strict JSON, tolerates missing keys, falls back to `semantic_text = text` with empty filters if the model output can't be parsed.

- [ ] **Step 1: Write the failing test (Ollama mocked via httpx MockTransport)**

```python
# tests/test_parser.py
import json
from datetime import date
import httpx
import pytest
from photosearch.query.parser import parse_query

def _client(payload):
    def handler(request):
        body = {"message": {"content": json.dumps(payload)}}
        return httpx.Response(200, json=body)
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))

async def test_parse_extracts_filters():
    payload = {"semantic_text": "dog at the beach", "date_from": "2023-06-01",
               "date_to": "2023-08-31", "location": "beach", "people": ["Mom"]}
    async with _client(payload) as c:
        f = await parse_query("my dog at the beach with Mom last summer",
                              "http://x", "qwen2.5:3b", client=c)
    assert f.semantic_text == "dog at the beach"
    assert f.date_from == date(2023, 6, 1) and f.date_to == date(2023, 8, 31)
    assert f.people == ["Mom"]

async def test_parse_falls_back_on_garbage():
    def handler(request):
        return httpx.Response(200, json={"message": {"content": "not json"}})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
        f = await parse_query("red car", "http://x", "m", client=c)
    assert f.semantic_text == "red car" and f.people == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_parser.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Write the implementation**

```python
# src/photosearch/query/parser.py
import json
from dataclasses import dataclass, field
from datetime import date

import httpx

_SYSTEM = (
    "You convert a photo search request into JSON with keys: "
    "semantic_text (string: the visual content to match), "
    "date_from, date_to (YYYY-MM-DD or null), location (string or null), "
    "people (array of names). Respond with ONLY the JSON object."
)


@dataclass
class QueryFilters:
    semantic_text: str
    date_from: date | None = None
    date_to: date | None = None
    location: str | None = None
    people: list[str] = field(default_factory=list)


def _parse_date(v):
    if not v:
        return None
    try:
        return date.fromisoformat(v)
    except (ValueError, TypeError):
        return None


async def parse_query(text: str, ollama_url: str, model: str,
                      client: httpx.AsyncClient | None = None) -> QueryFilters:
    owns = client is None
    client = client or httpx.AsyncClient()
    try:
        resp = await client.post(
            f"{ollama_url}/api/chat",
            json={
                "model": model,
                "format": "json",
                "stream": False,
                "messages": [
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": text},
                ],
            },
            timeout=60,
        )
        content = resp.json()["message"]["content"]
        data = json.loads(content)
    except Exception:
        return QueryFilters(semantic_text=text)
    finally:
        if owns:
            await client.aclose()

    return QueryFilters(
        semantic_text=data.get("semantic_text") or text,
        date_from=_parse_date(data.get("date_from")),
        date_to=_parse_date(data.get("date_to")),
        location=data.get("location") or None,
        people=[p for p in (data.get("people") or []) if isinstance(p, str)],
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_parser.py -v`
Expected: PASS.

- [ ] **Step 5: Checkpoint**

Run: `pytest`
Expected: all tests pass.

---

### Task 12: Search (sqlite-vec KNN + filters)

**Files:**
- Create: `src/photosearch/query/search.py`, `tests/test_search.py`

**Interfaces:**
- Consumes: `db` connection; `Embedder`; `QueryFilters`.
- Produces:
  - `@dataclass SearchResult` fields: `photo_id: str`, `path: str`, `thumb_path: str | None`, `taken_at: str | None`, `score: float`.
  - `def search(conn, embedder: Embedder, filters: QueryFilters, limit: int = 50) -> list[SearchResult]` — embeds `semantic_text`, runs a `photo_vectors` KNN, then filters the candidate set by date range and people (resolve names → `person_id` via `people`/`faces`). `score = 1 - distance`. People/date filters are applied as SQL `WHERE` on the candidate ids.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_search.py
import numpy as np
from datetime import date
from photosearch.query.search import search
from photosearch.query.parser import QueryFilters
from photosearch.db import PHOTO_DIM

class OneHotEmbedder:
    def embed_text(self, text):
        v = np.zeros(PHOTO_DIM, dtype="float32")
        v[0 if "dog" in text else 1] = 1.0
        return v
    def embed_image(self, img):
        raise NotImplementedError

def _add_photo(db, pid, vec, taken_at=None):
    db.execute("INSERT INTO photos(id, path, content_hash, taken_at, thumb_path) VALUES (?,?,?,?,?)",
               (pid, f"/{pid}.jpg", pid, taken_at, f"/{pid}.thumb.jpg"))
    db.execute("INSERT INTO photo_vectors(photo_id, embedding) VALUES (?,?)",
               (pid, vec.astype("float32").tobytes()))

def test_search_ranks_by_similarity(db):
    dog = np.zeros(PHOTO_DIM, dtype="float32"); dog[0] = 1.0
    cat = np.zeros(PHOTO_DIM, dtype="float32"); cat[1] = 1.0
    _add_photo(db, "dog1", dog); _add_photo(db, "cat1", cat)
    db.commit()
    results = search(db, OneHotEmbedder(), QueryFilters(semantic_text="a dog"), limit=10)
    assert results[0].photo_id == "dog1"

def test_search_date_filter(db):
    dog = np.zeros(PHOTO_DIM, dtype="float32"); dog[0] = 1.0
    _add_photo(db, "old", dog, taken_at="2020-01-01T00:00:00")
    _add_photo(db, "new", dog, taken_at="2024-06-01T00:00:00")
    db.commit()
    f = QueryFilters(semantic_text="dog", date_from=date(2024, 1, 1))
    ids = {r.photo_id for r in search(db, OneHotEmbedder(), f, limit=10)}
    assert ids == {"new"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_search.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Write the implementation**

```python
# src/photosearch/query/search.py
from dataclasses import dataclass

from ..embedder import Embedder
from .parser import QueryFilters


@dataclass
class SearchResult:
    photo_id: str
    path: str
    thumb_path: str | None
    taken_at: str | None
    score: float


def search(conn, embedder: Embedder, filters: QueryFilters, limit: int = 50) -> list[SearchResult]:
    qvec = embedder.embed_text(filters.semantic_text).astype("float32")
    # KNN over a widened candidate pool so post-filters still have results.
    k = max(limit * 5, 50)
    rows = conn.execute(
        """
        SELECT pv.photo_id AS pid, pv.distance AS dist,
               p.path AS path, p.thumb_path AS thumb, p.taken_at AS taken_at
        FROM photo_vectors pv
        JOIN photos p ON p.id = pv.photo_id
        WHERE pv.embedding MATCH ? AND k = ?
        ORDER BY pv.distance
        """,
        (qvec.tobytes(), k),
    ).fetchall()

    person_ids = _resolve_people(conn, filters.people)
    people_ok = _photos_with_people(conn, person_ids) if person_ids else None

    results: list[SearchResult] = []
    for r in rows:
        ta = r["taken_at"]
        if filters.date_from and (not ta or ta[:10] < filters.date_from.isoformat()):
            continue
        if filters.date_to and (not ta or ta[:10] > filters.date_to.isoformat()):
            continue
        if people_ok is not None and r["pid"] not in people_ok:
            continue
        results.append(SearchResult(r["pid"], r["path"], r["thumb"], ta, 1.0 - r["dist"]))
        if len(results) >= limit:
            break
    return results


def _resolve_people(conn, names: list[str]) -> list[int]:
    ids: list[int] = []
    for name in names:
        for row in conn.execute("SELECT id FROM people WHERE name = ? COLLATE NOCASE", (name,)):
            ids.append(row["id"])
    return ids


def _photos_with_people(conn, person_ids: list[int]) -> set[str]:
    if not person_ids:
        return set()
    q = ",".join("?" for _ in person_ids)
    rows = conn.execute(f"SELECT DISTINCT photo_id FROM faces WHERE person_id IN ({q})", person_ids)
    return {r["photo_id"] for r in rows}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_search.py -v`
Expected: PASS.

- [ ] **Step 5: Checkpoint**

Run: `pytest`
Expected: all tests pass.

---

### Task 13: FastAPI app + endpoints

**Files:**
- Create: `src/photosearch/api.py`, `tests/test_api.py`

**Interfaces:**
- Consumes: `config.Config`; `db`; `Embedder`; `parse_query`; `search`; serves `web/` static files.
- Produces:
  - `def create_app(conn, embedder: Embedder, config: Config) -> FastAPI`.
  - `GET /api/search?q=...` → `{"results": [SearchResult...]}` (runs `parse_query` then `search`).
  - `GET /api/thumb/{photo_id}` → the thumbnail JPEG (404 if missing).
  - `GET /api/people` → `[{"id", "name", "count"}]`.
  - `POST /api/people/{id}/name` body `{"name": str}` → 204; updates `people.name`.
  - `GET /api/status` → `{"photos": int, "faces": int, "people": int, "last_scan": str|null}`.
  - Static mount: `GET /` serves `web/index.html`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_api.py
import numpy as np
from fastapi.testclient import TestClient
from photosearch.api import create_app
from photosearch.config import Config
from photosearch.db import PHOTO_DIM
from pathlib import Path

def _config(tmp_path):
    return Config(share_paths=[], db_path=tmp_path / "d.db", thumb_dir=tmp_path / "t",
                  clip_model="m", clip_pretrained="p", ollama_url="http://x",
                  ollama_model="m", scan_interval_s=900)

class StubEmbedder:
    def embed_text(self, text):
        v = np.zeros(PHOTO_DIM, dtype="float32"); v[0] = 1.0; return v
    def embed_image(self, img): raise NotImplementedError

def test_status_and_people(db, tmp_path, monkeypatch):
    db.execute("INSERT INTO people(name) VALUES ('Mom')")
    db.commit()
    app = create_app(db, StubEmbedder(), _config(tmp_path))
    client = TestClient(app)
    assert client.get("/api/status").json()["people"] == 1
    pid = db.execute("SELECT id FROM people").fetchone()["id"]
    assert client.post(f"/api/people/{pid}/name", json={"name": "Mother"}).status_code == 204
    assert db.execute("SELECT name FROM people").fetchone()["name"] == "Mother"

def test_search_endpoint(db, tmp_path, monkeypatch):
    dog = np.zeros(PHOTO_DIM, dtype="float32"); dog[0] = 1.0
    db.execute("INSERT INTO photos(id, path, content_hash, thumb_path) VALUES ('d','/d.jpg','d','/t/d.jpg')")
    db.execute("INSERT INTO photo_vectors(photo_id, embedding) VALUES ('d', ?)", (dog.tobytes(),))
    db.commit()
    # avoid real Ollama: stub parse_query
    import photosearch.api as api_mod
    async def fake_parse(text, url, model, client=None):
        from photosearch.query.parser import QueryFilters
        return QueryFilters(semantic_text=text)
    monkeypatch.setattr(api_mod, "parse_query", fake_parse)
    app = create_app(db, StubEmbedder(), _config(tmp_path))
    r = TestClient(app).get("/api/search", params={"q": "dog"})
    assert r.status_code == 200 and r.json()["results"][0]["photo_id"] == "d"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_api.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Write the implementation**

```python
# src/photosearch/api.py
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import Config
from .embedder import Embedder
from .query.parser import parse_query
from .query.search import search

_WEB = Path(__file__).resolve().parent.parent.parent / "web"


class NameBody(BaseModel):
    name: str


def create_app(conn, embedder: Embedder, config: Config) -> FastAPI:
    app = FastAPI(title="photosearch")

    @app.get("/api/search")
    async def api_search(q: str, limit: int = 50):
        filters = await parse_query(q, config.ollama_url, config.ollama_model)
        results = search(conn, embedder, filters, limit=limit)
        return {"results": [asdict(r) for r in results]}

    @app.get("/api/thumb/{photo_id}")
    def api_thumb(photo_id: str):
        row = conn.execute("SELECT thumb_path FROM photos WHERE id = ?", (photo_id,)).fetchone()
        if not row or not row["thumb_path"] or not Path(row["thumb_path"]).exists():
            raise HTTPException(404)
        return FileResponse(row["thumb_path"], media_type="image/jpeg")

    @app.get("/api/people")
    def api_people():
        rows = conn.execute(
            """SELECT pe.id AS id, pe.name AS name, count(f.id) AS count
               FROM people pe LEFT JOIN faces f ON f.person_id = pe.id
               GROUP BY pe.id ORDER BY count DESC"""
        )
        return [dict(r) for r in rows]

    @app.post("/api/people/{person_id}/name", status_code=204)
    def api_name(person_id: int, body: NameBody):
        conn.execute("UPDATE people SET name = ? WHERE id = ?", (body.name, person_id))
        conn.commit()

    @app.get("/api/status")
    def api_status():
        def count(t):
            return conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
        last = conn.execute("SELECT value FROM meta WHERE key = 'last_scan'").fetchone()
        return {"photos": count("photos"), "faces": count("faces"),
                "people": count("people"), "last_scan": last["value"] if last else None}

    if _WEB.exists():
        app.mount("/", StaticFiles(directory=_WEB, html=True), name="web")
    return app
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_api.py -v`
Expected: PASS.

- [ ] **Step 5: Checkpoint**

Run: `pytest`
Expected: all tests pass.

---

### Task 14: Web UI

**Files:**
- Create: `web/index.html`, `web/app.js`, `web/style.css`

**Interfaces:**
- Consumes: the `/api/*` endpoints from Task 13.
- Produces: a static single-page UI with a search box + result grid and a "People" panel to rename clusters. No build step (vanilla JS).

**Note:** this is static frontend; it is exercised by the manual check in Step 3, not unit tests (the API is already covered by Task 13).

- [ ] **Step 1: Write `web/index.html`**

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Photo Search</title>
  <link rel="stylesheet" href="/style.css" />
</head>
<body>
  <header>
    <h1>Photo Search</h1>
    <form id="search-form">
      <input id="q" type="search" placeholder="e.g. my dog at the beach with Mom last summer" autofocus />
      <button type="submit">Search</button>
    </form>
    <div id="status"></div>
  </header>
  <main>
    <section id="results" class="grid"></section>
    <aside id="people"></aside>
  </main>
  <script src="/app.js"></script>
</body>
</html>
```

- [ ] **Step 2: Write `web/app.js`**

```javascript
const $ = (sel) => document.querySelector(sel);

async function loadStatus() {
  const s = await (await fetch("/api/status")).json();
  $("#status").textContent =
    `${s.photos} photos · ${s.faces} faces · ${s.people} people` +
    (s.last_scan ? ` · last scan ${s.last_scan}` : "");
}

async function loadPeople() {
  const people = await (await fetch("/api/people")).json();
  $("#people").innerHTML = "<h2>People</h2>";
  for (const p of people) {
    const row = document.createElement("div");
    row.className = "person";
    const input = document.createElement("input");
    input.value = p.name || "";
    input.placeholder = `Unnamed (${p.count})`;
    input.addEventListener("change", async () => {
      await fetch(`/api/people/${p.id}/name`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: input.value }),
      });
    });
    row.appendChild(input);
    $("#people").appendChild(row);
  }
}

async function doSearch(q) {
  const res = await (await fetch(`/api/search?q=${encodeURIComponent(q)}`)).json();
  const grid = $("#results");
  grid.innerHTML = "";
  for (const r of res.results) {
    const img = document.createElement("img");
    img.src = `/api/thumb/${r.photo_id}`;
    img.title = `${r.path}\nscore ${r.score.toFixed(3)}`;
    img.loading = "lazy";
    grid.appendChild(img);
  }
  if (!res.results.length) grid.textContent = "No matches.";
}

$("#search-form").addEventListener("submit", (e) => {
  e.preventDefault();
  doSearch($("#q").value.trim());
});

loadStatus();
loadPeople();
```

- [ ] **Step 3: Write `web/style.css` and manually verify**

```css
* { box-sizing: border-box; }
body { font-family: system-ui, sans-serif; margin: 0; color: #1a1a1a; }
header { padding: 1rem; border-bottom: 1px solid #ddd; }
h1 { margin: 0 0 .5rem; font-size: 1.25rem; }
#search-form { display: flex; gap: .5rem; }
#q { flex: 1; padding: .5rem; font-size: 1rem; }
#status { color: #666; font-size: .85rem; margin-top: .5rem; }
main { display: flex; gap: 1rem; padding: 1rem; align-items: flex-start; }
.grid { flex: 1; display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr)); gap: .5rem; }
.grid img { width: 100%; aspect-ratio: 1; object-fit: cover; border-radius: 6px; background: #eee; }
#people { width: 220px; }
.person input { width: 100%; padding: .35rem; margin-bottom: .35rem; }
```

Manual check: with the service running (Task 15) and at least one ingested photo, open `http://localhost:8756`, run a query, confirm the result grid renders thumbnails and a person rename persists after reload.

- [ ] **Step 4: Checkpoint**

Run: `pytest`
Expected: all tests pass (frontend adds no unit tests).

---

### Task 15: Background service + launchd

**Files:**
- Create: `src/photosearch/service.py`, `tests/test_service.py`
- Create: `com.photosearch.plist` (launchd template)

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `def run_scan_cycle(conn, config, embedder, detector) -> int` — one pass: `scan_for_changes` → `ingest_photo` for each → `cluster_faces` → write `meta.last_scan`. Returns number of photos ingested. Testable with fakes.
  - `def main() -> None` — the `photosearch` console entrypoint: loads config (path from `PHOTOSEARCH_CONFIG` env or `~/.photosearch/config.toml`), builds real `ClipEmbedder` + `InsightFaceDetector`, starts the scan loop in a background thread, and runs uvicorn on port 8756.

- [ ] **Step 1: Write the failing test (scan cycle only; `main` is not unit-tested)**

```python
# tests/test_service.py
from datetime import datetime, timezone
from PIL import Image
from photosearch.service import run_scan_cycle
from photosearch.config import Config

def _cfg(tmp_path, share):
    return Config(share_paths=[share], db_path=tmp_path / "d.db", thumb_dir=tmp_path / "t",
                  clip_model="m", clip_pretrained="p", ollama_url="http://x",
                  ollama_model="m", scan_interval_s=900)

def test_run_scan_cycle_ingests_and_records(db, tmp_path, fake_embedder, fake_detector):
    share = tmp_path / "share"; share.mkdir()
    Image.new("RGB", (40, 40), "red").save(share / "a.jpg")
    Image.new("RGB", (40, 40), "blue").save(share / "b.jpg")
    n = run_scan_cycle(db, _cfg(tmp_path, share), fake_embedder, fake_detector)
    assert n == 2
    assert db.execute("SELECT count(*) FROM photos").fetchone()[0] == 2
    last = db.execute("SELECT value FROM meta WHERE key='last_scan'").fetchone()
    assert last is not None
    # second cycle ingests nothing new
    assert run_scan_cycle(db, _cfg(tmp_path, share), fake_embedder, fake_detector) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_service.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Write the implementation**

```python
# src/photosearch/service.py
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import uvicorn

from .clustering import cluster_faces
from .config import load_config
from .pipeline import ingest_photo
from .scanner import scan_for_changes

PORT = 8756


def run_scan_cycle(conn, config, embedder, detector) -> int:
    ingested = 0
    for path in scan_for_changes(conn, config.share_paths):
        if ingest_photo(path, conn, embedder, detector, config.thumb_dir) is not None:
            ingested += 1
    if ingested:
        cluster_faces(conn)
    conn.execute(
        "INSERT OR REPLACE INTO meta(key, value) VALUES ('last_scan', ?)",
        (datetime.now(timezone.utc).isoformat(timespec="seconds"),),
    )
    conn.commit()
    return ingested


def _scan_loop(config, embedder, detector):
    from .db import connect, init_schema
    conn = connect(config.db_path)
    init_schema(conn)
    while True:
        try:
            run_scan_cycle(conn, config, embedder, detector)
        except Exception as e:  # keep the loop alive; log and retry next interval
            print(f"[scan] error: {e}")
        time.sleep(config.scan_interval_s)


def main() -> None:
    cfg_path = Path(os.environ.get("PHOTOSEARCH_CONFIG", Path.home() / ".photosearch" / "config.toml"))
    config = load_config(cfg_path)

    from .db import connect, init_schema
    from .embedder import ClipEmbedder
    from .faces import InsightFaceDetector
    from .api import create_app

    conn = connect(config.db_path)
    init_schema(conn)
    embedder = ClipEmbedder(config.clip_model, config.clip_pretrained)
    detector = InsightFaceDetector()

    threading.Thread(target=_scan_loop, args=(config, embedder, detector), daemon=True).start()
    app = create_app(conn, embedder, config)
    uvicorn.run(app, host="127.0.0.1", port=PORT)


if __name__ == "__main__":
    main()
```

Note: the scan loop uses its **own** SQLite connection (SQLite connections are not thread-safe to share). The API's connection is separate; both point at the same file.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_service.py -v`
Expected: PASS.

- [ ] **Step 5: Write the launchd template**

```xml
<!-- com.photosearch.plist -> copy to ~/Library/LaunchAgents/ and edit paths -->
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.photosearch</string>
  <key>ProgramArguments</key>
  <array>
    <string>/ABSOLUTE/PATH/TO/venv/bin/photosearch</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PHOTOSEARCH_CONFIG</key><string>/Users/YOU/.photosearch/config.toml</string>
  </dict>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>/Users/YOU/.photosearch/out.log</string>
  <key>StandardErrorPath</key><string>/Users/YOU/.photosearch/err.log</string>
</dict>
</plist>
```

Manual install (documented, not automated): copy the plist to `~/Library/LaunchAgents/`, edit the absolute paths, then `launchctl load ~/Library/LaunchAgents/com.photosearch.plist`. Confirm `http://localhost:8756` responds and Ollama is installed and running with the configured model pulled (`brew install ollama && ollama serve &` then `ollama pull qwen2.5:3b`). Ollama runs natively (not in a container) so it keeps Metal acceleration. Unit tests mock Ollama, so it is not required until runtime/integration.

- [ ] **Step 6: Checkpoint**

Run: `pytest`
Expected: all tests pass. Then run `pytest -m integration` once (with models + Ollama available) to smoke-test the real embedder and detector.

---

## Post-Implementation

- Run `pytest -m integration` with MobileCLIP, InsightFace, and Ollama available to verify real inference end to end.
- Manual end-to-end: point `config.toml` at a small NAS folder, start the service, wait for one scan cycle, then search and rename a person in the UI.
