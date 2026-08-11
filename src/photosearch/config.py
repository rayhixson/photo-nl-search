import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    share_paths: list[Path]
    db_path: Path
    thumb_dir: Path
    clip_model: str
    clip_pretrained: str
    ollama_url: str
    ollama_model: str
    scan_interval_s: int


def _p(value: str) -> Path:
    return Path(value).expanduser()


def load_config(path: Path) -> Config:
    data = tomllib.load(path.open("rb"))
    return Config(
        share_paths=[_p(s) for s in data["share_paths"]],
        db_path=_p(data["db_path"]),
        thumb_dir=_p(data["thumb_dir"]),
        clip_model=data["clip_model"],
        clip_pretrained=data["clip_pretrained"],
        ollama_url=data["ollama_url"],
        ollama_model=data["ollama_model"],
        scan_interval_s=int(data["scan_interval_s"]),
    )
