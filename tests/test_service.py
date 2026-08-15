from datetime import datetime, timezone
from PIL import Image
from photosearch.service import run_scan_cycle
from photosearch.config import Config


def _cfg(tmp_path, share):
    return Config(share_paths=[share], db_path=tmp_path / "d.db", thumb_dir=tmp_path / "t",
                  clip_model="m", clip_pretrained="p", ollama_url="http://x",
                  ollama_model="m", scan_interval_s=900)


def test_run_scan_cycle_ingests_and_records(db, tmp_path, fake_embedder, fake_detector):
    share = tmp_path / "share"; share.mkdir()
    Image.new("RGB", (40, 40), "red").save(share / "a.jpg")
    Image.new("RGB", (40, 40), "blue").save(share / "b.jpg")
    n = run_scan_cycle(db, _cfg(tmp_path, share), fake_embedder, fake_detector)
    assert n == 2
    assert db.execute("SELECT count(*) FROM photos").fetchone()[0] == 2
    last = db.execute("SELECT value FROM meta WHERE key='last_scan'").fetchone()
    assert last is not None
    # second cycle ingests nothing new
    assert run_scan_cycle(db, _cfg(tmp_path, share), fake_embedder, fake_detector) == 0
