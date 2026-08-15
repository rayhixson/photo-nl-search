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
