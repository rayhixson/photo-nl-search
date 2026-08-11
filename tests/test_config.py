from pathlib import Path
from photosearch.config import load_config

def test_load_config_expands_and_parses(tmp_path):
    cfg_file = tmp_path / "c.toml"
    cfg_file.write_text(
        'share_paths = ["/Volumes/nas"]\n'
        'db_path = "~/.photosearch/photos.db"\n'
        'thumb_dir = "~/.photosearch/thumbs"\n'
        'clip_model = "MobileCLIP-S1"\n'
        'clip_pretrained = "datacompdr"\n'
        'ollama_url = "http://localhost:11434"\n'
        'ollama_model = "qwen2.5:3b"\n'
        'scan_interval_s = 900\n'
    )
    cfg = load_config(cfg_file)
    assert cfg.share_paths == [Path("/Volumes/nas")]
    assert cfg.db_path == Path.home() / ".photosearch" / "photos.db"
    assert cfg.scan_interval_s == 900
    assert cfg.clip_model == "MobileCLIP-S1"
