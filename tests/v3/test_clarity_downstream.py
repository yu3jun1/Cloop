from __future__ import annotations

import pytest
import torch

from cloop.v1.world import EnsembleWorldModel, PersistenceDynamics
from cloop.v3.clarity_downstream import (
    _rows_for_task,
    load_downstream_config,
    prepare_rows,
)
from cloop.v3.clarity_downstream_head import (
    ClarityDownstreamError,
    ClarityOutcomeAdapter,
    cox_breslow_nll,
    fixed_time_survival_targets,
    outcome_loss,
    summarize_member_logits,
)
from cloop.v3.clarity_downstream_metrics import evaluate_cohort, selective_ipcw
from cloop.v3.clarity_survival_module import SurvivalModule
from cloop.v3.next_stage import _refs


def _head_kwargs():
    return {
        "hidden_dim": 8,
        "attention_dim": 8,
        "num_twoway_layers": 2,
        "num_heads": 2,
        "dropout": 0.0,
    }


def test_adapter_matches_fixed_upstream_module():
    torch.manual_seed(4)
    upstream = SurvivalModule(
        latent_dim=6, num_modalities=1, condition_dim=5, **_head_kwargs(),
    ).eval()
    adapter = ClarityOutcomeAdapter(6, 5, **_head_kwargs()).eval()
    adapter.head.load_state_dict(upstream.state_dict())
    pre = torch.randn(3, 6)
    post = torch.randn(3, 6)
    condition = torch.randn(3, 5)
    expected = upstream(pre[:, None, :], post[:, None, :], condition)
    actual = adapter(pre, post, condition)
    assert torch.allclose(actual[0], expected[0].squeeze(-1))
    assert torch.allclose(actual[1], expected[1].squeeze(-1))
    with pytest.raises(ClarityDownstreamError):
        adapter(pre[:, None], post, condition)


def test_cox_breslow_is_stable_and_has_correct_direction():
    time = torch.tensor([10.0, 20.0, 30.0, 30.0])
    event = torch.tensor([1, 1, 1, 0])
    good = torch.tensor([3.0, 2.0, 1.0, 0.0], requires_grad=True)
    bad = torch.flip(good.detach(), dims=(0,))
    assert cox_breslow_nll(good, time, event) < cox_breslow_nll(bad, time, event)
    assert torch.allclose(
        cox_breslow_nll(good + 1000.0, time, event),
        cox_breslow_nll(good, time, event),
        atol=1e-5,
    )
    perm = torch.tensor([0, 1, 3, 2])
    assert torch.allclose(
        cox_breslow_nll(good, time, event),
        cox_breslow_nll(good[perm], time[perm], event[perm]),
    )
    loss = cox_breslow_nll(good * 100.0, time, event)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(good.grad).all()
    no_event = cox_breslow_nll(good, time, torch.zeros_like(event))
    assert no_event == 0


def test_fixed_time_labels_follow_clarity_mask_and_both_heads_get_gradients():
    time = torch.tensor([100.0, 100.0, 500.0, 365.0])
    event = torch.tensor([0, 1, 1, 0])
    labels, valid = fixed_time_survival_targets(time, event, 365.0)
    assert labels.tolist() == [0.0, 0.0, 1.0, 0.0]
    assert valid.tolist() == [False, True, True, False]
    risk = torch.tensor([0.1, 0.4, -0.2, 0.0], requires_grad=True)
    logit = torch.tensor([0.1, -0.4, 0.2, 0.0], requires_grad=True)
    result = outcome_loss(risk, logit, time, event)
    result["total"].backward()
    assert risk.grad is not None and torch.isfinite(risk.grad).all()
    assert logit.grad is not None and torch.isfinite(logit.grad).all()


def test_member_probability_aggregation_does_not_average_logits():
    risk = torch.tensor([[1.0], [2.0]])
    logits = torch.tensor([[-4.0], [1.0]])
    summary = summarize_member_logits(risk, logits)
    assert summary["member_survival"].shape == (2, 1)
    assert not torch.allclose(
        summary["mean_survival"], torch.sigmoid(logits.mean(0)),
    )
    single = summarize_member_logits(risk[:1], logits[:1])
    assert single["probability_std"].item() == 0.0


def test_prepare_rows_keeps_member_horizon_and_token_axes_separate(tiny):
    _, bundle = tiny
    refs = _refs(bundle, "validation", 1, labels=True)
    world = EnsembleWorldModel([PersistenceDynamics(), PersistenceDynamics()])
    rows = prepare_rows(
        world, bundle, "validation", refs, 8,
        variant="ensemble", fold=0, seed=7, device=torch.device("cpu"),
    )
    assert rows
    row = rows[0]
    assert row["pre"].shape == (bundle.latent_dim,)
    assert row["post_members"].shape == (2, bundle.latent_dim)
    assert row["condition"].shape == (
        2 * bundle.clinical_dim + bundle.history_dim + 1,
    )
    assert torch.allclose(row["post_mean"], row["pre"])
    assert row["target_index"] - row["start_index"] == 1
    assert row["time"] > 0
    assert row["event"] in (0, 1)


def test_metrics_wire_raw_risk_to_cindex_and_probability_to_brier():
    reference = [
        {"time": 100.0, "event": 1},
        {"time": 500.0, "event": 0},
        {"time": 700.0, "event": 1},
    ]
    rows = [
        {
            "time": 100.0, "event": 1, "risk_score": 2.0,
            "survival_logit365": -2.0, "survival365": float(torch.sigmoid(torch.tensor(-2.0))),
            "latent_mse": 0.3,
        },
        {
            "time": 500.0, "event": 0, "risk_score": -1.0,
            "survival_logit365": 2.0, "survival365": float(torch.sigmoid(torch.tensor(2.0))),
            "latent_mse": 0.2,
        },
    ]
    result = evaluate_cohort(rows, reference, 365.0)
    assert result["c_index_risk"]["value"] == 1.0
    assert result["ipcw_brier365"]["value"] is not None
    assert result["td_auc365"]["value"] == 1.0


def test_selective_ranking_keeps_early_censored_patients_in_denominator():
    reference = [
        {"time": 100.0, "event": 1},
        {"time": 500.0, "event": 0},
        {"time": 800.0, "event": 0},
    ]
    rows = [
        {
            "record_id": f"r{i}", "time": time, "event": event,
            "survival365_mean_probability": probability,
            "probability_std": uncertainty,
        }
        for i, (time, event, probability, uncertainty) in enumerate([
            (50.0, 0, 0.5, 0.1),
            (100.0, 1, 0.2, 0.2),
            (500.0, 0, 0.9, 0.3),
        ])
    ]
    curve = selective_ipcw(
        rows, reference, [1.0, 2 / 3], 7, random_reference_repeats=3,
    )
    assert curve[0]["total"] == 3
    assert curve[0]["retained"] == 3
    assert curve[1]["retained"] == 2
    assert curve[1]["actual_coverage"] == pytest.approx(2 / 3)


def test_config_isolated_and_uses_fixed_clarity_head():
    spec = load_downstream_config("configs/v3/clarity_downstream.yaml")
    assert spec["run"] == "next_stage_clarity_outcome_v1"
    assert spec["protocol"] == "development_reuse"
    assert spec["variants"] == ["baseline", "rrt", "ensemble", "rrt_ensemble"]
    assert spec["head"]["num_twoway_layers"] == 2
    assert spec["train_outcome_independently"] is True
    assert spec["main_aggregation"] == "mean_latent"
    assert "value_weight" not in spec["training"]



def test_original_holdout_uses_train_validation_test_without_overlap():
    def row(patient_id):
        return {"patient_id": patient_id}

    cache = {
        "rows": {
            "baseline/fold0/seed7/H2/train": [row("fit")],
            "baseline/fold0/seed7/H2/validation": [row("stop")],
            "baseline/fold0/seed7/H2/test": [row("report")],
        },
    }
    run = {
        "protocol": "original_holdout",
        "patient_splits": {
            "fold0": {
                "fit_ids": ["fit"],
                "stop_ids": ["stop"],
                "report_ids": ["report"],
            },
        },
    }
    fit, stop, report = _rows_for_task(cache, run, "baseline", 0, 7, 2)
    assert [item["patient_id"] for item in fit] == ["fit"]
    assert [item["patient_id"] for item in stop] == ["stop"]
    assert [item["patient_id"] for item in report] == ["report"]
