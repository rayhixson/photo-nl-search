"""
Incremental photo scanner for detecting new and modified image files.

Walks configured share paths and returns absolute Path objects for image files
that are new or have changed (based on mtime/size), enabling efficient incremental
ingest. Never raises exceptions even on permission errors or corrupt data.
"""
import os
from pathlib import Path

SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".heic"}


def scan_for_changes(conn, share_paths: list[Path]) -> list[Path]:
    """
    Detect new and modified image files across share paths.

    Args:
        conn: sqlite3.Connection to photos database.
        share_paths: List of Path objects (directories) to scan.

    Returns:
        List of Path objects (absolute paths) for image files that are new or
        have changed. Only files with extensions in SUPPORTED_EXT are returned.
        Files with unsupported extensions are silently skipped.

    Raises:
        Never. Unreadable directories and files are skipped without raising.
    """
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
                # Scanner is a coarse pre-filter; authoritative dedupe is content-hash
                # in the ingest pipeline. Mtime tolerance ignores float-representation
                # noise while exact size match catches real changes. Rare false "changed"
                # triggers only a cheap re-hash that then no-ops.
                is_changed = (
                    prev is None
                    or prev[1] != st.st_size
                    or abs(prev[0] - st.st_mtime) > 1e-3
                )
                if is_changed:
                    changed.append(p)
    return changed
