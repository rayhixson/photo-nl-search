from typing import Protocol

import numpy as np
import open_clip
import torch
from PIL import Image

from .db import PHOTO_DIM


class Embedder(Protocol):
    def embed_image(self, img: Image.Image) -> np.ndarray: ...
    def embed_text(self, text: str) -> np.ndarray: ...


def _normalize(v: np.ndarray) -> np.ndarray:
    v = v.astype("float32")
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


class ClipEmbedder:
    def __init__(self, model_name: str, pretrained: str, device: str = "mps"):
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model = self.model.to(device).eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)
        assert self.model.visual.output_dim == PHOTO_DIM, "model dim != PHOTO_DIM"

    @torch.no_grad()
    def embed_image(self, img: Image.Image) -> np.ndarray:
        x = self.preprocess(img.convert("RGB")).unsqueeze(0).to(self.device)
        feats = self.model.encode_image(x)[0].cpu().numpy()
        return _normalize(feats)

    @torch.no_grad()
    def embed_text(self, text: str) -> np.ndarray:
        toks = self.tokenizer([text]).to(self.device)
        feats = self.model.encode_text(toks)[0].cpu().numpy()
        return _normalize(feats)
