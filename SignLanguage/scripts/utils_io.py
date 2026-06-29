from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import yaml


@dataclass
class Config:
    raw: dict

    @property
    def dataset_root(self) -> Path:
        return Path(self.raw["dataset"]["root"])

    @property
    def dictionary_path(self) -> Path:
        return Path(self.raw["dataset"]["dictionary"])

    @property
    def artifacts_dir(self) -> Path:
        return Path(self.raw["paths"]["artifacts_dir"])

    @property
    def extracted_dir(self) -> Path:
        return Path(self.raw["paths"]["extracted_features_dir"])

    @property
    def checkpoints_dir(self) -> Path:
        return Path(self.raw["paths"]["checkpoints_dir"])

    @property
    def default_ckpt(self) -> Path:
        return Path(self.raw["paths"]["default_checkpoint"])


def load_config(path: str | Path) -> Config:
    p = Path(path)
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    return Config(raw=data)


def load_dictionary(path: str | Path) -> Tuple[Dict[str, str], List[str]]:
    """
    返回：
    - id_to_word: {"000000": "情况", ...}
    - ids_sorted: ["000000", "000001", ...]（用于 class index）
    """
    p = Path(path)
    id_to_word: Dict[str, str] = {}
    ids: List[str] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        cid, word = parts[0], parts[1]
        id_to_word[cid] = word
        ids.append(cid)
    ids_sorted = sorted(ids)
    return id_to_word, ids_sorted

