# tests/test_search.py
import numpy as np
from datetime import date
from photosearch.query.search import search
from photosearch.query.parser import QueryFilters
from photosearch.db import PHOTO_DIM

class OneHotEmbedder:
    def embed_text(self, text):
        v = np.zeros(PHOTO_DIM, dtype="float32")
        v[0 if "dog" in text else 1] = 1.0
        return v
    def embed_image(self, img):
        raise NotImplementedError

def _add_photo(db, pid, vec, taken_at=None):
    db.execute("INSERT INTO photos(id, path, content_hash, taken_at, thumb_path) VALUES (?,?,?,?,?)",
               (pid, f"/{pid}.jpg", pid, taken_at, f"/{pid}.thumb.jpg"))
    db.execute("INSERT INTO photo_vectors(photo_id, embedding) VALUES (?,?)",
               (pid, vec.astype("float32").tobytes()))

def test_search_ranks_by_similarity(db):
    dog = np.zeros(PHOTO_DIM, dtype="float32"); dog[0] = 1.0
    cat = np.zeros(PHOTO_DIM, dtype="float32"); cat[1] = 1.0
    _add_photo(db, "dog1", dog); _add_photo(db, "cat1", cat)
    db.commit()
    results = search(db, OneHotEmbedder(), QueryFilters(semantic_text="a dog"), limit=10)
    assert results[0].photo_id == "dog1"

def test_search_date_filter(db):
    dog = np.zeros(PHOTO_DIM, dtype="float32"); dog[0] = 1.0
    _add_photo(db, "old", dog, taken_at="2020-01-01T00:00:00")
    _add_photo(db, "new", dog, taken_at="2024-06-01T00:00:00")
    db.commit()
    f = QueryFilters(semantic_text="dog", date_from=date(2024, 1, 1))
    ids = {r.photo_id for r in search(db, OneHotEmbedder(), f, limit=10)}
    assert ids == {"new"}

def test_search_date_to_filter(db):
    dog = np.zeros(PHOTO_DIM, dtype="float32"); dog[0] = 1.0
    _add_photo(db, "before", dog, taken_at="2022-03-15T00:00:00")
    _add_photo(db, "after",  dog, taken_at="2025-09-01T00:00:00")
    db.commit()
    f = QueryFilters(semantic_text="dog", date_to=date(2023, 12, 31))
    ids = {r.photo_id for r in search(db, OneHotEmbedder(), f, limit=10)}
    assert ids == {"before"}

def test_search_people_unknown_name_returns_empty(db):
    dog = np.zeros(PHOTO_DIM, dtype="float32"); dog[0] = 1.0
    _add_photo(db, "p1", dog); _add_photo(db, "p2", dog)
    db.commit()
    f = QueryFilters(semantic_text="dog", people=["NoSuchPerson"])
    results = search(db, OneHotEmbedder(), f, limit=10)
    assert results == []

def test_search_people_known_person_filters_correctly(db):
    dog = np.zeros(PHOTO_DIM, dtype="float32"); dog[0] = 1.0
    _add_photo(db, "alice_photo", dog)
    _add_photo(db, "other_photo", dog)
    db.commit()
    # Insert person and link alice_photo via a face row.
    db.execute("INSERT INTO people(id, name) VALUES (1, 'Alice')")
    db.execute("INSERT INTO faces(id, photo_id, person_id) VALUES ('f1', 'alice_photo', 1)")
    db.commit()
    f = QueryFilters(semantic_text="dog", people=["Alice"])
    ids = {r.photo_id for r in search(db, OneHotEmbedder(), f, limit=10)}
    assert ids == {"alice_photo"}
