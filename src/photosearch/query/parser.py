import json
from dataclasses import dataclass, field
from datetime import date

import httpx

_SYSTEM = (
    "You convert a photo search request into JSON with keys: "
    "semantic_text (string: the visual content to match), "
    "date_from, date_to (YYYY-MM-DD or null), location (string or null), "
    "people (array of names). Respond with ONLY the JSON object."
)


@dataclass
class QueryFilters:
    semantic_text: str
    date_from: date | None = None
    date_to: date | None = None
    location: str | None = None
    people: list[str] = field(default_factory=list)


def _parse_date(v):
    if not v:
        return None
    try:
        return date.fromisoformat(v)
    except (ValueError, TypeError):
        return None


async def parse_query(text: str, ollama_url: str, model: str,
                      client: httpx.AsyncClient | None = None) -> QueryFilters:
    owns = client is None
    client = client or httpx.AsyncClient()
    try:
        resp = await client.post(
            f"{ollama_url}/api/chat",
            json={
                "model": model,
                "format": "json",
                "stream": False,
                "messages": [
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": text},
                ],
            },
            timeout=60,
        )
        content = resp.json()["message"]["content"]
        data = json.loads(content)
    except Exception:
        return QueryFilters(semantic_text=text)
    finally:
        if owns:
            await client.aclose()

    return QueryFilters(
        semantic_text=data.get("semantic_text") or text,
        date_from=_parse_date(data.get("date_from")),
        date_to=_parse_date(data.get("date_to")),
        location=data.get("location") or None,
        people=[p for p in (data.get("people") or []) if isinstance(p, str)],
    )
