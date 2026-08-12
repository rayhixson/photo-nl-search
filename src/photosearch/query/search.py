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


def search(
    conn,
    embedder: Embedder,
    filters: QueryFilters,
    limit: int = 50,
    min_score: float = 0.0,
) -> list[SearchResult]:
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

    # Resolve a people filter. Explicit names from the parser are a STRICT
    # filter: if none exist, the result is empty ("photos of Mom" with no Mom).
    # Otherwise, if the whole query is itself a known person's name, treat it
    # like clicking that person in the UI — a bare-name search == the button.
    if filters.people:
        people_ok = _photos_with_people(conn, _resolve_people(conn, filters.people))
    else:
        name_ids = _resolve_people(conn, [filters.semantic_text]) if filters.semantic_text else []
        people_ok = _photos_with_people(conn, name_ids) if name_ids else None

    results: list[SearchResult] = []
    for r in rows:
        # Embeddings are L2-normalized, so cosine sim = 1 - dist^2/2. Higher is
        # more relevant (1.0 = identical). Rows are distance-ascending.
        score = 1.0 - (r["dist"] ** 2) / 2.0
        ta = r["taken_at"]
        if filters.date_from and (not ta or ta[:10] < filters.date_from.isoformat()):
            continue
        if filters.date_to and (not ta or ta[:10] > filters.date_to.isoformat()):
            continue
        if people_ok is not None:
            # Scoped to a person: include all their photos regardless of the
            # semantic score (the person match IS the relevance signal).
            if r["pid"] not in people_ok:
                continue
        elif score < min_score:
            # Pure semantic query: stop at the relevance floor (rows sorted).
            break
        results.append(SearchResult(r["pid"], r["path"], r["thumb"], ta, score))
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
