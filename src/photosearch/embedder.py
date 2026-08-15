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
        # Probe the actual output dim rather than reading model.visual.output_dim,
        # which timm-backed visual towers (e.g. MobileCLIP) do not expose.
        dim = self.embed_text("dimension probe").shape[0]
        assert dim == PHOTO_DIM, f"model embed dim {dim} != PHOTO_DIM {PHOTO_DIM}"

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
