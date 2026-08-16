import numpy as np
from fastapi.testclient import TestClient
from photosearch.api import create_app
from photosearch.config import Config
from photosearch.db import PHOTO_DIM
from pathlib import Path


def _config(tmp_path):
    return Config(share_paths=[], db_path=tmp_path / "d.db", thumb_dir=tmp_path / "t",
                  clip_model="m", clip_pretrained="p", ollama_url="http://x",
                  ollama_model="m", scan_interval_s=900)


class StubEmbedder:
    def embed_text(self, text):
        v = np.zeros(PHOTO_DIM, dtype="float32"); v[0] = 1.0; return v
    def embed_image(self, img): raise NotImplementedError


def test_status_and_people(db, tmp_path, monkeypatch):
    db.execute("INSERT INTO people(name) VALUES ('Mom')")
    db.commit()
    app = create_app(db, StubEmbedder(), _config(tmp_path))
    client = TestClient(app)
    assert client.get("/api/status").json()["people"] == 1
    pid = db.execute("SELECT id FROM people").fetchone()["id"]
    assert client.post(f"/api/people/{pid}/name", json={"name": "Mother"}).status_code == 204
    assert db.execute("SELECT name FROM people").fetchone()["name"] == "Mother"


def test_search_endpoint(db, tmp_path, monkeypatch):
    dog = np.zeros(PHOTO_DIM, dtype="float32"); dog[0] = 1.0
    db.execute("INSERT INTO photos(id, path, content_hash, thumb_path) VALUES ('d','/d.jpg','d','/t/d.jpg')")
    db.execute("INSERT INTO photo_vectors(photo_id, embedding) VALUES ('d', ?)", (dog.tobytes(),))
    db.commit()
    # avoid real Ollama: stub parse_query
    import photosearch.api as api_mod
    async def fake_parse(text, url, model, client=None):
        from photosearch.query.parser import QueryFilters
        return QueryFilters(semantic_text=text)
    monkeypatch.setattr(api_mod, "parse_query", fake_parse)
    app = create_app(db, StubEmbedder(), _config(tmp_path))
    r = TestClient(app).get("/api/search", params={"q": "dog"})
    assert r.status_code == 200 and r.json()["results"][0]["photo_id"] == "d"


def test_person_photos_endpoint(db, tmp_path):
    db.execute("INSERT INTO people(id, name) VALUES (1, 'Ray')")
    for pid in ("pA", "pB"):
        db.execute("INSERT INTO photos(id, path, content_hash, thumb_path) VALUES (?,?,?,?)",
                   (pid, f"/{pid}.jpg", pid, f"/t/{pid}.jpg"))
    db.execute("INSERT INTO faces(id, photo_id, person_id) VALUES ('f1','pA',1)")
    db.commit()
    app = create_app(db, StubEmbedder(), _config(tmp_path))
    r = TestClient(app).get("/api/people/1/photos")
    assert r.status_code == 200
    assert {x["photo_id"] for x in r.json()["results"]} == {"pA"}


def test_empty_name_clears_to_null(db, tmp_path):
    db.execute("INSERT INTO people(id, name) VALUES (1, 'Ray')")
    db.commit()
    app = create_app(db, StubEmbedder(), _config(tmp_path))
    assert TestClient(app).post("/api/people/1/name", json={"name": "   "}).status_code == 204
    assert db.execute("SELECT name FROM people WHERE id=1").fetchone()["name"] is None


def test_photos_browse_all(db, tmp_path):
    for pid in ("a", "b"):
        db.execute("INSERT INTO photos(id, path, content_hash, thumb_path) VALUES (?,?,?,?)",
                   (pid, f"/{pid}.jpg", pid, f"/t/{pid}.jpg"))
    db.commit()
    app = create_app(db, StubEmbedder(), _config(tmp_path))
    r = TestClient(app).get("/api/photos")
    assert r.status_code == 200
    assert {x["photo_id"] for x in r.json()["results"]} == {"a", "b"}


def test_photo_original_served(db, tmp_path):
    orig = tmp_path / "orig.jpg"; orig.write_bytes(b"\xff\xd8\xff original-bytes")
    db.execute("INSERT INTO photos(id, path, content_hash, thumb_path) VALUES ('o', ?, 'o', '/t/o.jpg')",
               (str(orig),))
    db.commit()
    app = create_app(db, StubEmbedder(), _config(tmp_path))
    client = TestClient(app)
    r = client.get("/api/photo/o")
    assert r.status_code == 200 and r.content == b"\xff\xd8\xff original-bytes"
    assert client.get("/api/photo/missing").status_code == 404


def test_reveal_in_finder(db, tmp_path, monkeypatch):
    orig = tmp_path / "orig.jpg"; orig.write_bytes(b"x")
    db.execute("INSERT INTO photos(id, path, content_hash, thumb_path) VALUES ('o', ?, 'o', '/t/o.jpg')",
               (str(orig),))
    db.commit()
    calls = []
    import photosearch.api as api_mod
    monkeypatch.setattr(api_mod.subprocess, "run", lambda *a, **k: calls.append(a[0]))
    app = create_app(db, StubEmbedder(), _config(tmp_path))
    client = TestClient(app)
    assert client.post("/api/photo/o/reveal").status_code == 204
    assert calls == [["open", "-R", str(orig)]]
    assert client.post("/api/photo/missing/reveal").status_code == 404  # 404 before subprocess


def test_photos_pagination(db, tmp_path):
    for i in range(3):
        pid = f"p{i}"
        db.execute("INSERT INTO photos(id, path, content_hash, thumb_path, taken_at) VALUES (?,?,?,?,?)",
                   (pid, f"/{pid}.jpg", pid, f"/t/{pid}.jpg", f"2024-01-0{i+1}T00:00:00"))
    db.commit()
    app = create_app(db, StubEmbedder(), _config(tmp_path))
    c = TestClient(app)
    r1 = c.get("/api/photos", params={"limit": 2, "offset": 0}).json()
    assert r1["total"] == 3 and len(r1["results"]) == 2 and r1["offset"] == 0 and r1["limit"] == 2
    r2 = c.get("/api/photos", params={"limit": 2, "offset": 2}).json()
    assert r2["total"] == 3 and len(r2["results"]) == 1
    ids1 = {x["photo_id"] for x in r1["results"]}
    ids2 = {x["photo_id"] for x in r2["results"]}
    assert ids1.isdisjoint(ids2) and len(ids1 | ids2) == 3  # no overlap, full coverage


def test_search_paginates_with_total(db, tmp_path, monkeypatch):
    dog = np.zeros(PHOTO_DIM, dtype="float32"); dog[0] = 1.0
    for i in range(2):
        db.execute("INSERT INTO photos(id, path, content_hash, thumb_path) VALUES (?,?,?,?)",
                   (f"d{i}", f"/d{i}.jpg", f"d{i}", f"/t/d{i}.jpg"))
        db.execute("INSERT INTO photo_vectors(photo_id, embedding) VALUES (?, ?)", (f"d{i}", dog.tobytes()))
    db.commit()
    import photosearch.api as api_mod
    async def fake_parse(text, url, model, client=None):
        from photosearch.query.parser import QueryFilters
        return QueryFilters(semantic_text=text)
    monkeypatch.setattr(api_mod, "parse_query", fake_parse)
    app = create_app(db, StubEmbedder(), _config(tmp_path))
    r = TestClient(app).get("/api/search", params={"q": "dog", "limit": 1, "offset": 0}).json()
    assert r["total"] == 2 and len(r["results"]) == 1 and r["limit"] == 1


def test_people_reports_faces_and_photo_counts(db, tmp_path):
    # Two faces in the SAME photo => faces=2 but photos=1 (the "17 vs 3" case).
    db.execute("INSERT INTO people(id, name) VALUES (1, 'Ray')")
    db.execute("INSERT INTO photos(id, path, content_hash, thumb_path) VALUES ('pA','/pA.jpg','pA','/t/pA.jpg')")
    db.execute("INSERT INTO faces(id, photo_id, person_id) VALUES ('f1','pA',1)")
    db.execute("INSERT INTO faces(id, photo_id, person_id) VALUES ('f2','pA',1)")
    db.commit()
    app = create_app(db, StubEmbedder(), _config(tmp_path))
    row = TestClient(app).get("/api/people").json()[0]
    assert row["faces"] == 2 and row["photos"] == 1
