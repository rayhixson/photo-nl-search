import pytest
from photosearch import db as _db


@pytest.fixture
def db(tmp_path):
    conn = _db.connect(tmp_path / "test.db")
    _db.init_schema(conn)
    yield conn
    conn.close()
