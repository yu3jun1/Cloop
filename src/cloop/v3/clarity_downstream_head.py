"""Cloop adapter and losses for the fixed CLARITY outcome architecture."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .clarity_survival_module import SurvivalModule


class ClarityDownstreamError(ValueError):
    pass


class ClarityOutcomeAdapter(nn.Module):
    """Adapt Cloop global latents ``[B,D]`` to CLARITY tokens ``[B,1,D]``."""

    def __init__(
        self,
        latent_dim: int,
        condition_dim: int,
        **head_config: Any,
    ) -> None:
        super().__init__()
        if int(latent_dim) < 1 or int(condition_dim) < 1:
            raise ClarityDownstreamError("latent_dim and condition_dim must be positive")
        self.latent_dim = int(latent_dim)
        self.condition_dim = int(condition_dim)
        self.head = SurvivalModule(
            latent_dim=self.latent_dim,
            num_modalities=1,
            condition_dim=self.condition_dim,
            **head_config,
        )

    def forward(self, pre: Tensor, post: Tensor, condition: Tensor) -> tuple[Tensor, Tensor]:
        if pre.ndim != 2 or post.shape != pre.shape or pre.shape[1] != self.latent_dim:
            raise ClarityDownstreamError("pre/post must be matching [B,D] tensors")
        if condition.ndim != 2 or condition.shape != (pre.shape[0], self.condition_dim):
            raise ClarityDownstreamError("condition must be [B,Ccond]")
        if not bool(
            torch.isfinite(pre).all()
            and torch.isfinite(post).all()
            and torch.isfinite(condition).all()
        ):
            raise ClarityDownstreamError("outcome inputs must be finite")
        risk, logit = self.head(pre[:, None, :], post[:, None, :], condition)
        return risk.squeeze(-1), logit.squeeze(-1)


def validate_survival_inputs(risk: Tensor, time: Tensor, event: Tensor) -> None:
    if risk.ndim != 1 or time.shape != risk.shape or event.shape != risk.shape:
        raise ClarityDownstreamError("risk/time/event must be matching [N] tensors")
    if risk.numel() == 0:
        raise ClarityDownstreamError("empty survival cohort")
    if not bool(torch.isfinite(risk).all() and torch.isfinite(time).all()):
        raise ClarityDownstreamError("non-finite risk/time")
    if not bool((time > 0).all()):
        raise ClarityDownstreamError("time must be positive")
    if not bool(((event == 0) | (event == 1)).all()):
        raise ClarityDownstreamError("event must be binary")


def cox_breslow_nll(risk: Tensor, time: Tensor, event: Tensor) -> Tensor:
    """Full-cohort Cox partial NLL with deterministic Breslow ties."""
    validate_survival_inputs(risk, time, event)
    observed = event.bool()
    if not bool(observed.any()):
        return risk.sum() * 0.0
    stable_risk = risk - risk.max()
    result = risk.sum() * 0.0
    for day in torch.unique(time[observed]):
        deaths = observed & (time == day)
        at_risk = time >= day
        result = result + deaths.sum() * torch.logsumexp(stable_risk[at_risk], dim=0)
        result = result - stable_risk[deaths].sum()
    return result / observed.sum()


def fixed_time_survival_targets(
    time: Tensor,
    event: Tensor,
    tau: float = 365.0,
) -> tuple[Tensor, Tensor]:
    if time.ndim != 1 or time.shape != event.shape or time.numel() == 0:
        raise ClarityDownstreamError("time/event must be non-empty matching [N] tensors")
    if not math.isfinite(tau) or tau <= 0:
        raise ClarityDownstreamError("tau must be finite and positive")
    if not bool(torch.isfinite(time).all() and (time > 0).all()):
        raise ClarityDownstreamError("invalid follow-up time")
    if not bool(((event == 0) | (event == 1)).all()):
        raise ClarityDownstreamError("event must be binary")
    labels = (time > tau).to(torch.float32)
    valid = (time > tau) | ((time <= tau) & (event == 1))
    return labels, valid


def outcome_loss(
    risk: Tensor,
    survival_logit: Tensor,
    time: Tensor,
    event: Tensor,
    tau: float = 365.0,
    w_cox: float = 0.5,
    w_bce: float = 0.5,
) -> dict[str, Tensor | int]:
    if survival_logit.shape != risk.shape:
        raise ClarityDownstreamError("risk and survival_logit must be matching [N] tensors")
    if not bool(torch.isfinite(survival_logit).all()):
        raise ClarityDownstreamError("non-finite survival_logit")
    if not (math.isfinite(w_cox) and math.isfinite(w_bce)) or min(w_cox, w_bce) < 0:
        raise ClarityDownstreamError("loss weights must be finite and non-negative")
    if w_cox + w_bce == 0:
        raise ClarityDownstreamError("at least one loss weight must be positive")
    cox = cox_breslow_nll(risk, time, event)
    labels, valid = fixed_time_survival_targets(time, event, tau)
    bce = (
        F.binary_cross_entropy_with_logits(survival_logit[valid], labels[valid])
        if bool(valid.any())
        else survival_logit.sum() * 0.0
    )
    return {
        "total": w_cox * cox + w_bce * bce,
        "cox_partial_nll": cox,
        "bce_identifiable": bce,
        "n_events": int(event.sum().item()),
        "n_identifiable": int(valid.sum().item()),
    }


def summarize_member_logits(
    member_risk: Tensor,
    member_survival_logit: Tensor,
) -> dict[str, Tensor]:
    """Summarize one shared head's outputs for dynamics members ``[M,B]``."""
    if member_risk.ndim != 2 or member_risk.shape != member_survival_logit.shape:
        raise ClarityDownstreamError("member outputs must be matching [M,B] tensors")
    if min(member_risk.shape) < 1:
        raise ClarityDownstreamError("empty ensemble/batch")
    if not bool(
        torch.isfinite(member_risk).all()
        and torch.isfinite(member_survival_logit).all()
    ):
        raise ClarityDownstreamError("non-finite member output")
    member_survival = torch.sigmoid(member_survival_logit)
    probability_variance = member_survival.var(0, unbiased=False)
    return {
        "member_survival": member_survival,
        "mean_survival": member_survival.mean(0),
        "mean_risk_score": member_risk.mean(0),
        "probability_variance": probability_variance,
        "probability_std": probability_variance.sqrt(),
    }
