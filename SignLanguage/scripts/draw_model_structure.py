from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# 允许直接用 `python scripts/xxx.py` 运行时也能导入项目根目录下的包
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.transformer_classifier import TransformerConfig
from scripts.utils_io import load_config, load_dictionary


def _add_box(ax, x: float, y: float, w: float, h: float, text: str) -> None:
    rect = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.4,
        edgecolor="#2a3f5f",
        facecolor="#e8f0ff",
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10)


def _add_arrow(ax, x1: float, y1: float, x2: float, y2: float) -> None:
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops={"arrowstyle": "->", "linewidth": 1.5, "color": "#334e73"},
    )


def draw_architecture(cfg: TransformerConfig, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 4.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    boxes = [
        (0.03, 0.28, 0.12, 0.44, f"Input Sequence\n(B, T, {cfg.input_dim})"),
        (0.20, 0.28, 0.12, 0.44, f"Linear Projection\n{cfg.input_dim} -> {cfg.d_model}"),
        (0.37, 0.28, 0.12, 0.44, "Positional Encoding"),
        (0.54, 0.28, 0.15, 0.44, f"Transformer Encoder\nLayers={cfg.num_layers}\nHeads={cfg.nhead}"),
        (0.74, 0.28, 0.10, 0.44, "CLS Token\nPooling"),
        (0.87, 0.28, 0.10, 0.44, f"Classifier Head\nnum_classes={cfg.num_classes}"),
    ]

    for x, y, w, h, text in boxes:
        _add_box(ax, x, y, w, h, text)

    _add_arrow(ax, 0.15, 0.5, 0.20, 0.5)
    _add_arrow(ax, 0.32, 0.5, 0.37, 0.5)
    _add_arrow(ax, 0.49, 0.5, 0.54, 0.5)
    _add_arrow(ax, 0.69, 0.5, 0.74, 0.5)
    _add_arrow(ax, 0.84, 0.5, 0.87, 0.5)

    ax.text(
        0.03,
        0.12,
        (
            "Model: TransformerClassifier | "
            f"d_model={cfg.d_model}, dim_feedforward={cfg.dim_feedforward}, dropout={cfg.dropout}"
        ),
        fontsize=10,
        color="#22324a",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    cfg = load_config(args.config)
    _, ids_sorted = load_dictionary(cfg.dictionary_path)
    model_cfg = TransformerConfig(
        input_dim=int(cfg.raw["features"]["per_frame_dim"]),
        num_classes=len(ids_sorted),
        max_seq_len=int(cfg.raw["train"]["max_seq_len"]),
        d_model=int(cfg.raw["model"]["d_model"]),
        nhead=int(cfg.raw["model"]["nhead"]),
        num_layers=int(cfg.raw["model"]["num_layers"]),
        dim_feedforward=int(cfg.raw["model"]["dim_feedforward"]),
        dropout=float(cfg.raw["model"]["dropout"]),
    )

    out_path = Path(args.out) if args.out else cfg.artifacts_dir / "figures" / "model_structure.png"
    draw_architecture(model_cfg, out_path=out_path)
    print(f"saved model structure figure to {out_path}")


if __name__ == "__main__":
    main()
