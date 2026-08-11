# Photo Natural-Language Search — Design Spec

**Date:** 2026-08-11
**Status:** Draft for review

## Goal

A fully-local Mac application that indexes photos stored on NAS network shares and
answers natural-language queries ("my dog at the beach last summer", "photos of Mom")
through a local web UI. All inference runs on-device — no cloud calls.

## Confirmed Decisions

| Choice | Decision |
|---|---|
| Target hardware | Mac (Apple Silicon) |
| Image embedding model | MobileCLIP |
| Query understanding | Hybrid — local LLM parses structured filters, then semantic vector search |
| Interface | Local web UI (browser) served by a local API |
| People / face search | Yes — face detection, clustering, and naming |
| Media scope | Photos only: JPEG / PNG / HEIC |
| Local LLM runtime | Ollama (small model, e.g. Qwen2.5 3B) |
| Deployment | Native launch-at-login background service (launchd) |
| Environment | Native `uv`-managed venv pinned to Python 3.12 — **no Docker**, to preserve Metal/MPS acceleration for MobileCLIP, InsightFace, and Ollama |
| Database | SQLite (single file) + `sqlite-vec` extension for vectors |

## Architecture

### Components

- **NAS connector** — reads from one or more SMB shares mounted read-only. User
  configures the root path(s) to index.
- **Scanner** — periodic incremental directory walk. SMB file-change events are
  unreliable, so the scanner polls on an interval and detects new/changed files by
  `path + mtime + size`. Work is enqueued and ingestion is idempotent, keyed by a
  content hash so re-scans never duplicate.
- **Ingestion workers** — per photo:
  1. Decode (Pillow + `pillow-heif` for HEIC).
  2. Extract EXIF: `taken_at`, GPS lat/lng, camera, orientation.
  3. Generate and cache a thumbnail on local disk.
  4. Compute a **MobileCLIP** image embedding.
  5. Detect faces and compute face embeddings (**InsightFace** via ONNX Runtime).
- **Face clustering** — batch job that clusters face vectors (HDBSCAN) into people.
  Clusters start unnamed; the user names them in the UI. A named person maps to a
  `person_id` used at query time.
- **Query service** — NL query flow:
  1. Query text → **Ollama** with a structured-output prompt → JSON
     `{ semantic_text, date_range, location, people[] }`.
  2. `semantic_text` → MobileCLIP **text** embedding (shared image/text space).
  3. Build a single SQL query: a `sqlite-vec` KNN search on the semantic embedding,
     joined with metadata filters (date range, GPS proximity, `person_id in [...]`)
     across the `photos`, `faces`, and `people` tables.
  4. Merge and rank; return image ids + thumbnails.
- **API + Web UI** — FastAPI backend serving a simple browser SPA: search box,
  result grid, and a people-naming screen.
- **Runtime** — the FastAPI app and the scan loop run as a launch-at-login service
  (launchd / `brew services`). Ollama runs as its own local service.

### Storage

A single **SQLite** file holds everything — relational data and vectors — via the
`sqlite-vec` extension. Tables:

- `photos` — `id`, `path`, `content_hash`, `taken_at`, `gps_lat`, `gps_lng`,
  `camera`, `width`, `height`, `thumb_path`.
- `photo_vectors` — `sqlite-vec` virtual table: `photo_id` → MobileCLIP embedding.
- `faces` — `id`, `photo_id`, `bbox`, `person_id`.
- `face_vectors` — `sqlite-vec` virtual table: `face_id` → face embedding.
- `people` — `id`, `name` (low-churn; populated when the user names a cluster).
- `meta` — one-row key/value for scan state (e.g. last-scan-time).

**Thumbnail cache** — thumbnails are stored as files on local disk (not in the DB),
so browsing results does not repeatedly read from the NAS.

Rationale: at a personal-library scale everything fits comfortably in one SQLite
file. Joins across `photos`/`faces`/`people` answer queries like "photos of Mom in
2024" in a single statement, backup is a file copy, and the `sqlite3` driver ships
with Python.

Setup caveat: `sqlite-vec` is a loadable extension, which requires SQLite
extension-loading to be enabled. macOS's system Python sometimes disables it —
resolve with `pip install pysqlite3-binary` or a Homebrew Python build.

Scale note: `sqlite-vec` performs brute-force KNN (no ANN index). This is fast for
tens of thousands of photos plus their face vectors; revisit an ANN-indexed store
only if the library grows into the hundreds of thousands.

### Data Flow

```
scan → queue → ingest workers → SQLite (photos, faces, vectors)
                                      │
                             periodic face clustering
                                      │
   API query: Ollama parse → MobileCLIP text embed → SQL query (sqlite-vec KNN + filters)
                                      │
                                  web UI grid
```

## Error Handling

- **NAS unreachable** → pause scanning, retry with backoff. Never crash the service.
- **Unreadable / corrupt image** → skip and log; continue the batch.
- **Model load failure** (MobileCLIP / InsightFace / Ollama unreachable) → fail fast
  at startup with a clear, actionable message.
- **Idempotent ingest** → keyed by content hash; re-scans and moved files do not
  create duplicates.

## Testing

- **Unit**
  - EXIF parsing (dates, GPS, missing/garbage tags).
  - Query parser: mock Ollama output, assert the derived filter JSON.
  - Thumbnail generation and orientation handling.
  - Incremental-scan change detection (new / modified / unchanged / deleted).
- **Integration**
  - Small fixture photo set → full ingest → assert search results for known queries.
  - Face clustering on a tiny labeled set → assert cluster/person assignment.
- The LLM is mocked in tests for deterministic results.

## Model Notes

- **MobileCLIP** runs via `open_clip` on PyTorch with the MPS (Apple GPU) backend.
  Its shared text/image embedding space means the same model embeds both photos and
  query text. CoreML export is a later speed optimization, not required for v1.
- **InsightFace** (e.g. `buffalo_l`) runs on ONNX Runtime for face detection +
  embedding.
- **Ollama** serves a small instruct model for filter extraction; the prompt requests
  strict JSON and the parser tolerates/repairs minor deviations.

## Out of Scope (v1)

- RAW and video formats.
- Cloud sync or remote access beyond the home network.
- Editing/tagging photos on the NAS (indexing is read-only).
