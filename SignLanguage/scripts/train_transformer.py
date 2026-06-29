from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

# 允许直接用 `python scripts/xxx.py` 运行时也能导入项目根目录下的包
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.transformer_classifier import TransformerClassifier, TransformerConfig
from scripts.utils_io import load_config, load_dictionary


def save_training_plots(history: List[Dict[str, float]], out_path: Path) -> None:
    if not history:
        return

    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    train_acc = [h["train_acc"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    val_acc = [h["val_acc"] for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, train_loss, label="train_loss", marker="o")
    if any(np.isfinite(v) for v in val_loss):
        axes[0].plot(epochs, val_loss, label="val_loss", marker="o")
    axes[0].set_title("Loss Curve")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, train_acc, label="train_acc", marker="o")
    if any(np.isfinite(v) for v in val_acc):
        axes[1].plot(epochs, val_acc, label="val_acc", marker="o")
    axes[1].set_title("Accuracy Curve")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


class FeatureDataset(Dataset):
    def __init__(self, index: List[Dict]) -> None:
        self.index = index

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        item = self.index[i]
        x = np.load(item["x_path"]).astype(np.float32)  # (T, D)
        mask = np.load(item["mask_path"]).astype(np.bool_)  # (T,)
        y = np.int64(item["label"])
        return torch.from_numpy(x), torch.from_numpy(mask), torch.tensor(y, dtype=torch.long)


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = torch.argmax(logits, dim=-1)
    return float((pred == y).float().mean().item())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--num_epochs", type=int, default=50, help="训练轮数，默认 50")
    ap.add_argument("--early_stop_patience", type=int, default=10, help="早停容忍轮数（验证集无提升时）")
    ap.add_argument("--early_stop_min_delta", type=float, default=1e-4, help="验证集提升最小阈值")
    ap.add_argument("--lr_patience", type=int, default=4, help="学习率调度耐心轮数")
    ap.add_argument("--lr_factor", type=float, default=0.5, help="学习率衰减倍率")
    ap.add_argument("--min_lr", type=float, default=1e-6, help="学习率下限")
    args = ap.parse_args()

    cfg = load_config(args.config)
    _, ids_sorted = load_dictionary(cfg.dictionary_path)

    out_root = cfg.extracted_dir
    index_path = out_root / "index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"未找到特征索引：{index_path}。请先运行 scripts/extract_mediapipe_features.py")

    index = json.loads(index_path.read_text(encoding="utf-8"))
    if len(index) == 0:
        raise RuntimeError("特征索引为空，可能提取失败或目录无样本。")

    ds = FeatureDataset(index)

    seed = int(cfg.raw["train"]["seed"])
    g = torch.Generator().manual_seed(seed)
    n_total = len(ds)
    if n_total < 2:
        ds_train = ds
        ds_val = None
    else:
        n_val = max(1, int(0.1 * n_total))
        n_train = n_total - n_val
        ds_train, ds_val = random_split(ds, [n_train, n_val], generator=g)

    bs = int(cfg.raw["train"]["batch_size"])
    num_workers = int(cfg.raw["train"]["num_workers"])
    dl_train = DataLoader(ds_train, batch_size=bs, shuffle=True, num_workers=num_workers)
    dl_val = None if ds_val is None else DataLoader(ds_val, batch_size=bs, shuffle=False, num_workers=num_workers)

    input_dim = int(cfg.raw["features"]["per_frame_dim"])
    num_classes = len(ids_sorted)
    mcfg = TransformerConfig(
        input_dim=input_dim,
        num_classes=num_classes,
        max_seq_len=int(cfg.raw["train"]["max_seq_len"]),
        d_model=int(cfg.raw["model"]["d_model"]),
        nhead=int(cfg.raw["model"]["nhead"]),
        num_layers=int(cfg.raw["model"]["num_layers"]),
        dim_feedforward=int(cfg.raw["model"]["dim_feedforward"]),
        dropout=float(cfg.raw["model"]["dropout"]),
    )

    device = torch.device(args.device)
    model = TransformerClassifier(mcfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.raw["train"]["lr"]), weight_decay=float(cfg.raw["train"]["weight_decay"]))
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="max",
        factor=float(args.lr_factor),
        patience=int(args.lr_patience),
        min_lr=float(args.min_lr),
    )

    cfg.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = cfg.artifacts_dir / "figures"
    best_acc = -1.0
    best_saved = False
    history: List[Dict[str, float]] = []
    early_stop_counter = 0

    num_epochs = int(args.num_epochs)
    for epoch in range(1, num_epochs + 1):
        model.train()
        tr_loss = 0.0
        tr_acc = 0.0
        n = 0
        for x, mask, y in dl_train:
            x = x.to(device)
            mask = mask.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x, key_padding_mask=mask)
            loss = crit(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            bsz = x.size(0)
            tr_loss += float(loss.item()) * bsz
            tr_acc += accuracy(logits.detach(), y) * bsz
            n += bsz

        tr_loss /= max(1, n)
        tr_acc /= max(1, n)

        model.eval()
        va_loss = 0.0
        va_acc = 0.0
        n2 = 0
        if dl_val is not None:
            with torch.no_grad():
                for x, mask, y in dl_val:
                    x = x.to(device)
                    mask = mask.to(device)
                    y = y.to(device)
                    logits = model(x, key_padding_mask=mask)
                    loss = crit(logits, y)
                    bsz = x.size(0)
                    va_loss += float(loss.item()) * bsz
                    va_acc += accuracy(logits, y) * bsz
                    n2 += bsz
            va_loss /= max(1, n2)
            va_acc /= max(1, n2)
        else:
            va_loss = float("nan")
            va_acc = float("nan")

        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(tr_loss),
                "train_acc": float(tr_acc),
                "val_loss": float(va_loss),
                "val_acc": float(va_acc),
            }
        )

        ckpt = {
            "model": model.state_dict(),
            "config": asdict(mcfg),
            "epoch": epoch,
            "val_acc": va_acc,
        }
        last_path = cfg.checkpoints_dir / "last.pt"
        torch.save(ckpt, last_path)

        if dl_val is not None:
            improved = (va_acc - best_acc) > float(args.early_stop_min_delta)
            if improved:
                best_acc = va_acc
                best_path = cfg.checkpoints_dir / "best.pt"
                torch.save(ckpt, best_path)
                best_saved = True
                early_stop_counter = 0
            else:
                early_stop_counter += 1

        if dl_val is None:
            print(f"epoch={epoch} train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} (no val split; dataset too small)")
        else:
            scheduler.step(va_acc)
            current_lr = float(opt.param_groups[0]["lr"])
            print(
                f"epoch={epoch} train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} "
                f"val_loss={va_loss:.4f} val_acc={va_acc:.4f} best={best_acc:.4f} "
                f"lr={current_lr:.2e} es={early_stop_counter}/{args.early_stop_patience}"
            )

            if early_stop_counter >= int(args.early_stop_patience):
                print(f"early stopping at epoch={epoch}, best_val_acc={best_acc:.4f}")
                break

    if dl_val is None and not best_saved:
        # 仅用于极小数据调试场景：确保后端能加载到权重文件
        best_path = cfg.checkpoints_dir / "best.pt"
        last_path = cfg.checkpoints_dir / "last.pt"
        if last_path.exists():
            best_path.write_bytes(last_path.read_bytes())
            print(f"saved best checkpoint to {best_path}")

    history_path = figures_dir / "train_history.json"
    history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    save_training_plots(history=history, out_path=figures_dir / "training_curves.png")
    print(f"saved training history to {history_path}")
    print(f"saved training curves to {figures_dir / 'training_curves.png'}")


if __name__ == "__main__":
    main()

