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


def passes_filter(det_score: float, bbox: tuple[int, int, int, int],
                  min_score: float, min_size: int) -> bool:
    """Keep only confident, non-tiny detections. InsightFace fires on non-faces
    (headlights, murals, patterns) with low det_score, and on tiny background
    faces; both pollute clustering. Filtering them at the source keeps clusters
    clean."""
    x1, y1, x2, y2 = bbox
    return det_score >= min_score and (x2 - x1) >= min_size and (y2 - y1) >= min_size


class InsightFaceDetector:
    def __init__(self, model_name: str = "buffalo_l",
                 min_score: float = 0.6, min_size: int = 50):
        from insightface.app import FaceAnalysis

        self.app = FaceAnalysis(name=model_name)
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.min_score = min_score
        self.min_size = min_size

    def detect(self, img: Image.Image) -> list[Face]:
        rgb = np.asarray(img.convert("RGB"))
        bgr = rgb[:, :, ::-1]
        out: list[Face] = []
        for f in self.app.get(bgr):
            x1, y1, x2, y2 = (int(v) for v in f.bbox)
            det = float(getattr(f, "det_score", 1.0))
            if not passes_filter(det, (x1, y1, x2, y2), self.min_score, self.min_size):
                continue
            out.append(Face(bbox=(x1, y1, x2, y2), embedding=f.normed_embedding.astype("float32")))
        return out
