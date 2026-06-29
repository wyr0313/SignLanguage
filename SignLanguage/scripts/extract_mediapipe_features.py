from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

# 允许直接用 `python scripts/xxx.py` 运行时也能导入项目根目录下的包
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.mediapipe_handlandmarker import HandLandmarkerWrapper, result_to_feature
from scripts.utils_io import load_config, load_dictionary


def _list_samples_for_class(video_class_dir: Path) -> List[Path]:
    # video-png/000000/P01_01_00_0 (directory of frames)
    return sorted([p for p in video_class_dir.iterdir() if p.is_dir()])


def _read_frames(sample_dir: Path) -> List[Path]:
    # frames are typically png files; some datasets may be no extension, so include all files
    frames = [p for p in sample_dir.iterdir() if p.is_file()]
    frames_sorted = sorted(frames, key=lambda p: p.name)
    return frames_sorted


def extract_one_sample(
    landmarker: HandLandmarkerWrapper,
    sample_dir: Path,
    max_seq_len: int,
) -> Tuple[np.ndarray, np.ndarray]:
    frames = _read_frames(sample_dir)
    if len(frames) == 0:
        x = np.zeros((max_seq_len, 126), dtype=np.float32)
        mask = np.ones((max_seq_len,), dtype=np.bool_)  # all padding
        return x, mask

    xs: List[np.ndarray] = []
    for fp in frames[:max_seq_len]:
        img_bgr = cv2.imread(str(fp))
        if img_bgr is None:
            xs.append(np.zeros((126,), dtype=np.float32))
            continue
        res = landmarker.detect_bgr(img_bgr)
        feat = result_to_feature(res)
        xs.append(feat)

    x = np.stack(xs, axis=0).astype(np.float32)  # (T, 126)
    t = x.shape[0]
    if t < max_seq_len:
        pad = np.zeros((max_seq_len - t, x.shape[1]), dtype=np.float32)
        x = np.concatenate([x, pad], axis=0)
    mask = np.zeros((max_seq_len,), dtype=np.bool_)
    if t < max_seq_len:
        mask[t:] = True
    return x, mask


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--limit_classes", type=int, default=0, help="仅调试用：限制处理前 N 个类别（0 表示全部）")
    ap.add_argument("--limit_samples_per_class", type=int, default=0, help="仅调试用：限制每类前 N 个样本（0 表示全部）")
    args = ap.parse_args()

    cfg = load_config(args.config)
    id_to_word, ids_sorted = load_dictionary(cfg.dictionary_path)

    dataset_root = cfg.dataset_root
    video_root = dataset_root / "video-png"
    out_root = cfg.extracted_dir
    out_root.mkdir(parents=True, exist_ok=True)

    max_seq_len = int(cfg.raw["train"]["max_seq_len"])

    model_path = Path(cfg.raw["paths"]["hand_landmarker_task"])
    landmarker = HandLandmarkerWrapper.create(model_path=model_path, num_hands=int(cfg.raw["features"]["max_num_hands"]))

    index: List[Dict] = []

    class_ids = ids_sorted
    if args.limit_classes and args.limit_classes > 0:
        class_ids = class_ids[: args.limit_classes]

    for cls_id in class_ids:
        cls_dir = video_root / cls_id
        if not cls_dir.exists():
            continue
        samples = _list_samples_for_class(cls_dir)
        if args.limit_samples_per_class and args.limit_samples_per_class > 0:
            samples = samples[: args.limit_samples_per_class]

        cls_out = out_root / cls_id
        cls_out.mkdir(parents=True, exist_ok=True)

        y = class_ids.index(cls_id)  # class index in current run (debug-safe)
        for sample_dir in samples:
            sample_id = sample_dir.name
            x, mask = extract_one_sample(landmarker, sample_dir=sample_dir, max_seq_len=max_seq_len)
            npy_path = cls_out / f"{sample_id}.npy"
            m_path = cls_out / f"{sample_id}.mask.npy"
            np.save(npy_path, x)
            np.save(m_path, mask)
            index.append(
                {
                    "class_id": cls_id,
                    "word": id_to_word.get(cls_id, ""),
                    "label": int(y),
                    "sample_id": sample_id,
                    "x_path": str(npy_path.as_posix()),
                    "mask_path": str(m_path.as_posix()),
                }
            )

    (out_root / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

