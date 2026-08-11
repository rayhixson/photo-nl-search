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
