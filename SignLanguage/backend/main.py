from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketDisconnect, WebSocketState

# 允许直接用 `python backend/main.py` 运行时也能导入项目根目录下的包
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.transformer_classifier import TransformerClassifier, TransformerConfig
from scripts.mediapipe_handlandmarker import HandLandmarkerWrapper, result_to_feature
from scripts.utils_io import load_config, load_dictionary


def _decode_data_url_to_bgr(data_url: str) -> Optional[np.ndarray]:
    # expects "data:image/jpeg;base64,...."
    if "," not in data_url:
        return None
    _, b64 = data_url.split(",", 1)
    raw = base64.b64decode(b64)
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


class RealtimeBuffer:
    def __init__(self, max_len: int, feat_dim: int) -> None:
        self.max_len = max_len
        self.feat_dim = feat_dim
        self.x = np.zeros((max_len, feat_dim), dtype=np.float32)
        self.valid = 0

    def push(self, feat: np.ndarray) -> None:
        if feat.shape[0] != self.feat_dim:
            return
        if self.valid < self.max_len:
            self.x[self.valid] = feat
            self.valid += 1
        else:
            self.x[:-1] = self.x[1:]
            self.x[-1] = feat

    def as_tensors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # (1, T, D), mask True for padding
        x = self.x.copy()
        mask = np.ones((self.max_len,), dtype=np.bool_)
        if self.valid > 0:
            mask[: self.valid] = False
        return torch.from_numpy(x).unsqueeze(0), torch.from_numpy(mask).unsqueeze(0)

    def clear(self) -> None:
        self.x.fill(0.0)
        self.valid = 0


def load_model(ckpt_path: Path, input_dim: int, num_classes: int, max_seq_len: int) -> Optional[TransformerClassifier]:
    if not ckpt_path.exists():
        return None
    ckpt = torch.load(ckpt_path, map_location="cpu")
    mcfg_dict = ckpt.get("config") or {}
    mcfg = TransformerConfig(
        input_dim=int(mcfg_dict.get("input_dim", input_dim)),
        num_classes=int(mcfg_dict.get("num_classes", num_classes)),
        max_seq_len=int(mcfg_dict.get("max_seq_len", max_seq_len)),
        d_model=int(mcfg_dict.get("d_model", 256)),
        nhead=int(mcfg_dict.get("nhead", 8)),
        num_layers=int(mcfg_dict.get("num_layers", 4)),
        dim_feedforward=int(mcfg_dict.get("dim_feedforward", 512)),
        dropout=float(mcfg_dict.get("dropout", 0.1)),
    )
    model = TransformerClassifier(mcfg)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model


def _normalize_class_id(class_id: str) -> str:
    class_id = class_id.strip()
    if class_id.isdigit():
        return class_id.zfill(6)
    return class_id


def _normalize_sample_id(sample_id: str) -> str:
    sid = sample_id.strip().lower()
    if sid.endswith(".mat"):
        sid = sid[:-4]
    return sid


def _sorted_dir_names(path: Path) -> List[str]:
    if not path.exists() or not path.is_dir():
        return []
    return sorted([p.name for p in path.iterdir() if p.is_dir()])


def _sorted_file_names(path: Path, suffix: str = "") -> List[str]:
    if not path.exists() or not path.is_dir():
        return []
    out = []
    for p in path.iterdir():
        if not p.is_file():
            continue
        if suffix and p.suffix.lower() != suffix.lower():
            continue
        out.append(p.name)
    return sorted(out)


def create_app(config_path: str) -> FastAPI:
    cfg = load_config(config_path)
    id_to_word, ids_sorted = load_dictionary(cfg.dictionary_path)

    input_dim = int(cfg.raw["features"]["per_frame_dim"])
    max_seq_len = int(cfg.raw["train"]["max_seq_len"])
    num_classes = len(ids_sorted)

    model = load_model(cfg.default_ckpt, input_dim=input_dim, num_classes=num_classes, max_seq_len=max_seq_len)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model is not None:
        model.to(device)

    model_path = Path(cfg.raw["paths"]["hand_landmarker_task"])
    landmarker = HandLandmarkerWrapper.create(model_path=model_path, num_hands=int(cfg.raw["features"]["max_num_hands"]))

    app = FastAPI()
    frontend_dir = Path("frontend").resolve()
    artifacts_dir = Path("artifacts").resolve()
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")
    app.mount("/artifacts", StaticFiles(directory=str(artifacts_dir)), name="artifacts")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(str(frontend_dir / "index.html"))

    @app.get("/api/dataset/self_check")
    def dataset_self_check(class_id: str, sample_id: str) -> Dict[str, Any]:
        cid = _normalize_class_id(class_id)
        sid = _normalize_sample_id(sample_id)
        if not sid:
            return {"ok": False, "error": "sample_id 不能为空"}

        dataset_root = Path(cfg.raw["dataset"]["root"]).resolve()
        depth_root = dataset_root / "xf200_body_depth_mat"
        color_root = dataset_root / "xf200_body_color_txt" / "xf500_body_color_txt"
        joints_root = dataset_root / "slr200_words_joints" / "slr500_words_joints"
        video_root = dataset_root / "video-png"

        mat_class_dir = depth_root / cid
        mat_path = mat_class_dir / f"{sid}.mat"
        mat_exists = mat_path.exists()

        color_class_dir = color_root / cid
        joints_class_dir = joints_root / str(int(cid)) if cid.isdigit() else joints_root / cid
        video_class_dir = video_root / cid

        mat_files = _sorted_file_names(mat_class_dir, ".mat")
        color_files = _sorted_file_names(color_class_dir, ".txt")
        joints_files = _sorted_file_names(joints_class_dir, ".txt")
        video_dirs = _sorted_dir_names(video_class_dir)

        matched_index = None
        if sid.isdigit():
            try:
                s = int(sid)
                if s > 0:
                    matched_index = s - 1
            except ValueError:
                matched_index = None

        candidate = {
            "video_png_dir": None,
            "color_txt_file": None,
            "joints_txt_file": None,
        }
        if matched_index is not None:
            if matched_index < len(video_dirs):
                candidate["video_png_dir"] = video_dirs[matched_index]
            if matched_index < len(color_files):
                candidate["color_txt_file"] = color_files[matched_index]
            if matched_index < len(joints_files):
                candidate["joints_txt_file"] = joints_files[matched_index]

        cls_word = id_to_word.get(cid, "")
        return {
            "ok": True,
            "class_id": cid,
            "class_word": cls_word,
            "sample_id": sid,
            "mat_exists": mat_exists,
            "mat_path": str(mat_path) if mat_exists else "",
            "mat_size_bytes": int(mat_path.stat().st_size) if mat_exists else 0,
            "class_dirs_exist": {
                "depth_mat": mat_class_dir.exists(),
                "color_txt": color_class_dir.exists(),
                "joints_txt": joints_class_dir.exists(),
                "video_png": video_class_dir.exists(),
            },
            "class_file_counts": {
                "depth_mat": len(mat_files),
                "color_txt": len(color_files),
                "joints_txt": len(joints_files),
                "video_png_dirs": len(video_dirs),
            },
            "sample_mat_file": f"{sid}.mat",
            "candidate_multimodal_match": candidate,
        }

    @app.websocket("/ws")
    async def ws(websocket: WebSocket) -> None:
        await websocket.accept()
        buf = RealtimeBuffer(max_len=max_seq_len, feat_dim=input_dim)
        min_detected_frames = 6
        clear_after_no_hand_frames = 10
        detected_frames = 0
        no_hand_frames = 0
        try:
            while True:
                try:
                    msg = await websocket.receive_text()
                except WebSocketDisconnect:
                    break
                data = json.loads(msg)
                data_url = data.get("image")
                if not isinstance(data_url, str):
                    await websocket.send_text(json.dumps({"ok": False, "error": "missing image"}))
                    continue

                img_bgr = _decode_data_url_to_bgr(data_url)
                if img_bgr is None:
                    await websocket.send_text(json.dumps({"ok": False, "error": "bad image"}))
                    continue

                res = landmarker.detect_bgr(img_bgr)
                has_hand = bool(getattr(res, "hand_landmarks", None))
                if has_hand:
                    feat = result_to_feature(res)
                    buf.push(feat)
                    detected_frames += 1
                    no_hand_frames = 0
                else:
                    no_hand_frames += 1
                    detected_frames = 0
                    if no_hand_frames >= clear_after_no_hand_frames:
                        buf.clear()

                out: Dict[str, Any] = {"ok": True}
                if model is None:
                    out["text"] = "未加载模型：请先训练生成 artifacts/checkpoints/best.pt"
                    out["confidence"] = 0.0
                    out["top3"] = []
                elif not has_hand:
                    out["text"] = "未检测到手势"
                    out["confidence"] = 0.0
                    out["class_id"] = "-"
                    out["top3"] = []
                elif detected_frames < min_detected_frames:
                    out["text"] = "检测中..."
                    out["confidence"] = 0.0
                    out["class_id"] = "-"
                    out["top3"] = []
                else:
                    x_t, mask_t = buf.as_tensors()
                    x_t = x_t.to(device)
                    mask_t = mask_t.to(device)
                    with torch.no_grad():
                        logits = model(x_t, key_padding_mask=mask_t)[0]
                        prob = torch.softmax(logits, dim=-1)
                        conf, pred = torch.max(prob, dim=-1)
                        topk_conf, topk_idx = torch.topk(prob, k=min(3, prob.numel()), dim=-1)
                    top3 = []
                    for score, idx in zip(topk_conf.tolist(), topk_idx.tolist()):
                        cls_id_k = ids_sorted[int(idx)]
                        top3.append(
                            {
                                "class_id": cls_id_k,
                                "text": id_to_word.get(cls_id_k, cls_id_k),
                                "confidence": float(score),
                            }
                        )
                    conf_value = float(conf.item())
                    out["top3"] = top3
                    cls_id = ids_sorted[int(pred.item())]
                    out["text"] = id_to_word.get(cls_id, cls_id)
                    out["class_id"] = cls_id
                    out["confidence"] = conf_value

                await websocket.send_text(json.dumps(out, ensure_ascii=False))
        finally:
            # 客户端主动断开时可能已发送 close，避免二次 close 导致异常
            try:
                if websocket.client_state != WebSocketState.DISCONNECTED:
                    await websocket.close()
            except RuntimeError:
                pass

    return app


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    import uvicorn

    app = create_app(args.config)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

