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
    b_ids = {r["person_id"] for r in db.execute("SELECT person_id FROM faces WHERE id LIKE 'b%'")}
    assert len(b_ids) == 1 and None not in b_ids and b_ids != a_ids


def test_named_person_survives_reclustering(db):
    """Named people ('Alice') must keep their faces after any subsequent re-cluster."""
    base_a = np.zeros(FACE_DIM, dtype="float32"); base_a[0] = 1.0
    base_b = np.zeros(FACE_DIM, dtype="float32"); base_b[1] = 1.0
    rng = np.random.default_rng(42)

    # Insert two well-separated clusters
    for i in range(4):
        _add_face(db, f"a{i}", base_a + rng.normal(0, 0.01, FACE_DIM))
    for i in range(4):
        _add_face(db, f"b{i}", base_b + rng.normal(0, 0.01, FACE_DIM))
    db.commit()

    # First cluster run → 2 clusters
    n = cluster_faces(db, min_cluster_size=3)
    assert n == 2

    # Name the person who owns the "a" faces
    alice_pid = db.execute(
        "SELECT DISTINCT person_id FROM faces WHERE id LIKE 'a%'"
    ).fetchone()["person_id"]
    db.execute("UPDATE people SET name = 'Alice' WHERE id = ?", (alice_pid,))
    db.commit()

    # Add a new "a"-like face (simulates Alice appearing in a new photo)
    _add_face(db, "a_new", base_a + rng.normal(0, 0.01, FACE_DIM))
    db.commit()

    # Re-cluster (triggers on every ingest cycle)
    n2 = cluster_faces(db, min_cluster_size=3)
    assert n2 == 2

    # Alice's named person row must still exist
    alice_row = db.execute("SELECT id FROM people WHERE name = 'Alice'").fetchone()
    assert alice_row is not None, "Alice person row must survive re-clustering"

    # Same person_id (no new row was created for Alice)
    assert alice_row["id"] == alice_pid, "Alice must keep the same person_id after re-clustering"

    # ALL "a*" faces — including the newly inserted one — belong to Alice
    a_pids = {r["person_id"] for r in db.execute("SELECT person_id FROM faces WHERE id LIKE 'a%'")}
    assert a_pids == {alice_pid}, "All a* faces (including a_new) must belong to Alice"

    # "b*" faces belong to a distinct (unnamed) person
    b_pids = {r["person_id"] for r in db.execute("SELECT person_id FROM faces WHERE id LIKE 'b%'")}
    assert len(b_pids) == 1 and None not in b_pids
    assert b_pids != {alice_pid}, "b* faces must belong to a different person than Alice"
