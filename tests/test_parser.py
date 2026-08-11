import json
from datetime import date
import httpx
import pytest
from photosearch.query.parser import parse_query


def _client(payload):
    def handler(request):
        body = {"message": {"content": json.dumps(payload)}}
        return httpx.Response(200, json=body)
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_parse_extracts_filters():
    payload = {"semantic_text": "dog at the beach", "date_from": "2023-06-01",
               "date_to": "2023-08-31", "location": "beach", "people": ["Mom"]}
    async with _client(payload) as c:
        f = await parse_query("my dog at the beach with Mom last summer",
                              "http://x", "qwen2.5:3b", client=c)
    assert f.semantic_text == "dog at the beach"
    assert f.date_from == date(2023, 6, 1) and f.date_to == date(2023, 8, 31)
    assert f.people == ["Mom"]


async def test_parse_falls_back_on_garbage():
    def handler(request):
        return httpx.Response(200, json={"message": {"content": "not json"}})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
        f = await parse_query("red car", "http://x", "m", client=c)
    assert f.semantic_text == "red car" and f.people == []
