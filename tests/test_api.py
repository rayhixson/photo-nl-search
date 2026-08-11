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
