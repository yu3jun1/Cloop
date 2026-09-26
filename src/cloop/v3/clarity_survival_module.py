"""CLARITY's two-way-attention survival outcome module.

Source: ``Predictor/models/survival_module.py`` at CLARITY commit
``dadb82241a24f5ec5e4e4dc994e3116fd4a9da04`` (blob
``21e984ee04745348b8fc5151e4650a63616a4567``).

Copyright (c) 2026 Tianxingjian Ding. Distributed under the MIT License;
the upstream license is available at https://github.com/DingTianxingjian/CLARITY/blob/dadb82241a24f5ec5e4e4dc994e3116fd4a9da04/LICENSE.

The class definitions below preserve the fixed upstream implementation.  The
Cloop-specific single-global-token adaptation lives in
``clarity_downstream_head.py`` so every dynamics variant uses this exact model.
"""

import torch
import torch.nn as nn


class TwoWayCrossAttentionLayer(nn.Module):
    """Single pre-norm bidirectional cross-attention layer."""

    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()

        self.cross_attn_1to2 = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn_2to1 = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )

        self.ffn1 = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )
        self.ffn2 = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )

        self.norm_q1 = nn.LayerNorm(dim)
        self.norm_q2 = nn.LayerNorm(dim)
        self.norm_ffn1 = nn.LayerNorm(dim)
        self.norm_ffn2 = nn.LayerNorm(dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, seq1, seq2):
        attn_out1, _ = self.cross_attn_1to2(
            query=self.norm_q1(seq1), key=seq2, value=seq2
        )
        seq1 = seq1 + self.dropout(attn_out1)

        attn_out2, _ = self.cross_attn_2to1(
            query=self.norm_q2(seq2), key=seq1, value=seq1
        )
        seq2 = seq2 + self.dropout(attn_out2)

        seq1 = seq1 + self.dropout(self.ffn1(self.norm_ffn1(seq1)))
        seq2 = seq2 + self.dropout(self.ffn2(self.norm_ffn2(seq2)))

        return seq1, seq2


class TwoWayTransformer(nn.Module):
    """Stacked two-way attention followed by per-sequence normalization."""

    def __init__(self, dim, num_heads=8, num_layers=2, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers

        self.layers = nn.ModuleList([
            TwoWayCrossAttentionLayer(dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.final_norm_seq1 = nn.LayerNorm(dim)
        self.final_norm_seq2 = nn.LayerNorm(dim)

    def forward(self, seq1, seq2):
        for layer in self.layers:
            seq1, seq2 = layer(seq1, seq2)
        return self.final_norm_seq1(seq1), self.final_norm_seq2(seq2)


class SurvivalModule(nn.Module):
    """CLARITY risk/survival-logit head over pre and predicted latents."""

    def __init__(
        self,
        latent_dim: int = 767,
        num_modalities: int = 4,
        hidden_dim: int = 128,
        attention_dim: int = 128,
        num_twoway_layers: int = 1,
        num_heads: int = 4,
        dropout: float = 0.5,
        condition_dim: int = 0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_modalities = num_modalities

        assert attention_dim % num_heads == 0, (
            f"attention_dim ({attention_dim}) must be divisible by num_heads ({num_heads})"
        )

        self.input_proj = nn.Linear(latent_dim, attention_dim)

        self.two_way_attn = TwoWayTransformer(
            dim=attention_dim,
            num_heads=num_heads,
            num_layers=num_twoway_layers,
            dropout=dropout,
        )

        if condition_dim > 0:
            self.condition_proj = nn.Linear(condition_dim, attention_dim)
            fusion_input_dim = attention_dim * 4
        else:
            self.condition_proj = None
            fusion_input_dim = attention_dim * 3

        self.modality_fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.risk_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.survival_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, pre_latent, pred_latent, condition_emb=None):
        pre_proj = self.input_proj(pre_latent)
        pred_proj = self.input_proj(pred_latent)

        pre_enhanced, pred_enhanced = self.two_way_attn(pre_proj, pred_proj)

        delta = pred_enhanced - pre_enhanced
        parts = [
            pre_enhanced.mean(dim=1),
            pred_enhanced.mean(dim=1),
            delta.mean(dim=1),
        ]
        if self.condition_proj is not None and condition_emb is not None:
            parts.append(self.condition_proj(condition_emb))
        combined = torch.cat(parts, dim=1)

        fused = self.modality_fusion(combined)
        risk_score = self.risk_head(fused)
        survival_logit = self.survival_head(fused)
        return risk_score, survival_logit
