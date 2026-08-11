from dataclasses import dataclass
from typing import Protocol

import numpy as np
from PIL import Image

from .db import FACE_DIM


@dataclass
class Face:
    bbox: tuple[int, int, int, int]
    embedding: np.ndarray


class FaceDetector(Protocol):
    def detect(self, img: Image.Image) -> list[Face]: ...


class InsightFaceDetector:
    def __init__(self, model_name: str = "buffalo_l"):
        from insightface.app import FaceAnalysis

        self.app = FaceAnalysis(name=model_name)
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    def detect(self, img: Image.Image) -> list[Face]:
        rgb = np.asarray(img.convert("RGB"))
        bgr = rgb[:, :, ::-1]
        out: list[Face] = []
        for f in self.app.get(bgr):
            emb = f.normed_embedding.astype("float32")
            x1, y1, x2, y2 = (int(v) for v in f.bbox)
            out.append(Face(bbox=(x1, y1, x2, y2), embedding=emb))
        return out
