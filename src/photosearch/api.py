import threading
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import Config
from .embedder import Embedder
from .query.parser import parse_query
from .query.search import search

_WEB = Path(__file__).resolve().parent.parent.parent / "web"


class NameBody(BaseModel):
    name: str


def create_app(conn, embedder: Embedder, config: Config) -> FastAPI:
    app = FastAPI(title="photosearch")
    _db_lock = threading.Lock()

    @app.get("/api/search")
    async def api_search(q: str, limit: int = 50):
        filters = await parse_query(q, config.ollama_url, config.ollama_model)
        with _db_lock:
            results = search(conn, embedder, filters, limit=limit)
        return {"results": [asdict(r) for r in results]}

    @app.get("/api/thumb/{photo_id}")
    def api_thumb(photo_id: str):
        with _db_lock:
            row = conn.execute(
                "SELECT thumb_path FROM photos WHERE id = ?", (photo_id,)
            ).fetchone()
        if not row or not row["thumb_path"] or not Path(row["thumb_path"]).exists():
            raise HTTPException(404)
        return FileResponse(row["thumb_path"], media_type="image/jpeg")

    @app.get("/api/people")
    def api_people():
        with _db_lock:
            rows = conn.execute(
                """SELECT pe.id AS id, pe.name AS name, count(f.id) AS count
                   FROM people pe LEFT JOIN faces f ON f.person_id = pe.id
                   GROUP BY pe.id ORDER BY count DESC"""
            ).fetchall()
        return [dict(r) for r in rows]

    @app.post("/api/people/{person_id}/name", status_code=204)
    def api_name(person_id: int, body: NameBody):
        with _db_lock:
            conn.execute("UPDATE people SET name = ? WHERE id = ?", (body.name, person_id))
            conn.commit()

    @app.get("/api/status")
    def api_status():
        with _db_lock:
            def count(t):
                return conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
            last = conn.execute("SELECT value FROM meta WHERE key = 'last_scan'").fetchone()
            return {
                "photos": count("photos"),
                "faces": count("faces"),
                "people": count("people"),
                "last_scan": last["value"] if last else None,
            }

    if _WEB.exists():
        app.mount("/", StaticFiles(directory=_WEB, html=True), name="web")
    return app
