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
