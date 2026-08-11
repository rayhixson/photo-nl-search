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
