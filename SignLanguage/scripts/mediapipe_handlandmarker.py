from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)


def ensure_hand_landmarker_model(model_path: Path) -> Path:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if model_path.exists() and model_path.stat().st_size > 0:
        return model_path
    tmp = model_path.with_suffix(".tmp")
    urllib.request.urlretrieve(HAND_LANDMARKER_URL, tmp)
    tmp.replace(model_path)
    return model_path


@dataclass
class HandLandmarkerWrapper:
    landmarker: vision.HandLandmarker

    @staticmethod
    def create(model_path: Path, num_hands: int = 2) -> "HandLandmarkerWrapper":
        ensure_hand_landmarker_model(model_path)
        base_options = python.BaseOptions(model_asset_path=str(model_path))
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=num_hands,
        )
        landmarker = vision.HandLandmarker.create_from_options(options)
        return HandLandmarkerWrapper(landmarker=landmarker)

    def detect_bgr(self, img_bgr: np.ndarray) -> "vision.HandLandmarkerResult":
        img_rgb = img_bgr[..., ::-1].copy()
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        return self.landmarker.detect(mp_image)


def result_to_feature(result: "vision.HandLandmarkerResult") -> np.ndarray:
    """
    输出 per-frame 特征 (126,)：
    - 左手 21*(x,y,z) + 右手 21*(x,y,z)
    - handedness 若缺失则按检测顺序填充
    """
    feat = np.zeros((2, 21, 3), dtype=np.float32)
    if not result.hand_landmarks:
        return feat.reshape(-1)

    # handedness: list[list[Category]]
    for idx_det, lms in enumerate(result.hand_landmarks[:2]):
        idx_lr = idx_det
        if result.handedness and idx_det < len(result.handedness) and result.handedness[idx_det]:
            label = result.handedness[idx_det][0].category_name  # "Left" / "Right"
            idx_lr = 0 if label.lower() == "left" else 1

        for i, lm in enumerate(lms):
            feat[idx_lr, i, 0] = float(lm.x)
            feat[idx_lr, i, 1] = float(lm.y)
            feat[idx_lr, i, 2] = float(lm.z)

    return feat.reshape(-1)

