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
