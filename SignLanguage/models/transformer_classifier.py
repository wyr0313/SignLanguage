from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 2048) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, T, D)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


@dataclass
class TransformerConfig:
    input_dim: int
    num_classes: int
    max_seq_len: int = 128
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.1


class TransformerClassifier(nn.Module):
    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.input_proj = nn.Linear(cfg.input_dim, cfg.d_model)
        self.pos = PositionalEncoding(cfg.d_model, dropout=cfg.dropout, max_len=cfg.max_seq_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_layers)

        self.cls = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        nn.init.normal_(self.cls, std=0.02)

        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.num_classes)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (B, T, input_dim)
        key_padding_mask: (B, T) True 表示该位置是 padding（将被 attention mask）
        """
        b, t, _ = x.shape
        if t > self.cfg.max_seq_len:
            x = x[:, : self.cfg.max_seq_len]
            if key_padding_mask is not None:
                key_padding_mask = key_padding_mask[:, : self.cfg.max_seq_len]
            t = self.cfg.max_seq_len

        x = self.input_proj(x)
        x = self.pos(x)

        cls_tok = self.cls.expand(b, -1, -1)
        x = torch.cat([cls_tok, x], dim=1)  # (B, 1+T, D)

        if key_padding_mask is not None:
            cls_mask = torch.zeros((b, 1), dtype=torch.bool, device=key_padding_mask.device)
            key_padding_mask = torch.cat([cls_mask, key_padding_mask], dim=1)

        h = self.encoder(x, src_key_padding_mask=key_padding_mask)
        h0 = self.norm(h[:, 0])
        return self.head(h0)

