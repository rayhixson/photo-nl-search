# src/photosearch/clustering.py
import numpy as np
from sklearn.cluster import HDBSCAN

_SIMILARITY_THRESHOLD = 0.5  # minimum cosine similarity to re-use a named person


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def cluster_faces(conn, min_cluster_size: int = 3) -> int:
    """Cluster face embeddings and assign person_ids.

    Named people survive re-clustering: for each new cluster the centroid is
    matched against every existing named person's centroid (cosine similarity).
    If the best match exceeds _SIMILARITY_THRESHOLD the cluster is assigned to
    that named person's existing row; otherwise a new unnamed people row is
    created.  Each named person can be claimed by at most one cluster (the one
    with the highest similarity); ties are broken by similarity score.

    Returns the number of clusters found (faces with noise label -1 are
    left with person_id NULL).
    """
    rows = list(conn.execute("SELECT face_id, embedding FROM face_vectors"))
    if not rows or len(rows) < min_cluster_size:
        return 0

    face_ids = [r["face_id"] for r in rows]
    vecs = np.stack([np.frombuffer(r["embedding"], dtype="float32") for r in rows])

    # --- Capture named-person centroids BEFORE resetting anything ---
    # {person_id: centroid_vec (L2-normalized)}
    named_centroids: dict[int, np.ndarray] = {}
    for row in conn.execute("SELECT id FROM people WHERE name IS NOT NULL"):
        pid = row["id"]
        emb_rows = list(conn.execute(
            """SELECT fv.embedding FROM face_vectors fv
               JOIN faces f ON f.id = fv.face_id
               WHERE f.person_id = ?""",
            (pid,),
        ))
        if not emb_rows:
            continue  # named person with no faces: skip (no centroid to match)
        embs = np.stack([np.frombuffer(r["embedding"], dtype="float32") for r in emb_rows])
        named_centroids[pid] = _l2_normalize(embs.mean(axis=0))

    # --- Cluster ---
    labels = HDBSCAN(
        min_cluster_size=min_cluster_size, metric="euclidean", copy=True
    ).fit_predict(vecs)

    # --- Reset prior auto-assignments; keep named people rows intact ---
    conn.execute("UPDATE faces SET person_id = NULL")
    conn.execute("DELETE FROM people WHERE name IS NULL")

    # --- Build per-cluster member index ---
    cluster_indices: dict[int, list[int]] = {}
    for i, label in enumerate(labels):
        label = int(label)
        if label >= 0:
            cluster_indices.setdefault(label, []).append(i)

    # --- Match each cluster to the best named-person centroid (two-pass) ---
    # Pass 1: for each cluster compute (best_sim, named_person_id)
    cluster_best: dict[int, tuple[float, int]] = {}  # label -> (sim, pid)
    if named_centroids:
        named_ids = list(named_centroids.keys())
        named_mat = np.stack([named_centroids[pid] for pid in named_ids])  # (N_named, D)
        for label, idxs in cluster_indices.items():
            centroid = _l2_normalize(vecs[idxs].mean(axis=0))
            sims = named_mat @ centroid  # dot product of normalized vectors = cosine sim
            best_idx = int(np.argmax(sims))
            best_sim = float(sims[best_idx])
            if best_sim >= _SIMILARITY_THRESHOLD:
                cluster_best[label] = (best_sim, named_ids[best_idx])

    # Pass 2: resolve conflicts — each named person claimed by at most one cluster
    # (the cluster with the highest similarity wins)
    named_claim: dict[int, tuple[float, int]] = {}  # pid -> (sim, label)
    for label, (sim, pid) in cluster_best.items():
        if pid not in named_claim or sim > named_claim[pid][0]:
            named_claim[pid] = (sim, label)

    # Build final cluster -> person_id mapping (named claims first)
    cluster_to_person: dict[int, int] = {}
    for pid, (_sim, label) in named_claim.items():
        cluster_to_person[label] = pid

    # Remaining clusters get a new unnamed people row
    for label in cluster_indices:
        if label not in cluster_to_person:
            cur = conn.execute("INSERT INTO people(name) VALUES (NULL)")
            cluster_to_person[label] = cur.lastrowid

    # --- Assign person_ids to faces ---
    for fid, label in zip(face_ids, labels):
        label = int(label)
        if label < 0:
            continue
        conn.execute(
            "UPDATE faces SET person_id = ? WHERE id = ?",
            (cluster_to_person[label], fid),
        )

    conn.commit()
    return len(cluster_to_person)
