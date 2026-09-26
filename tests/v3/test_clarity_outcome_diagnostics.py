from __future__ import annotations

import copy

import pytest
import torch

from cloop.v3.clarity_downstream import _tensor_state_hash
from cloop.v3.clarity_outcome_diagnostics import (
    ClarityDiagnosticError,
    _audit_summary,
    _post_summary,
    audit_training_task,
    load_diagnostic_config,
    make_post_diagnostic_rows,
)


def _rows(n: int = 4):
    return [
        {
            "record_id": f"r{index}",
            "variant": "baseline",
            "fold": 0,
            "seed": 7,
            "horizon": 2,
            "patient_id": f"p{index}",
            "post_mean": torch.tensor([float(index), 1.0]),
            "true_post": torch.tensor([float(index) + 0.5, 2.0]),
            "pre": torch.tensor([0.0, 0.0]),
            "condition": torch.tensor([0.0]),
            "time": 100.0 + index,
            "event": index % 2,
            "latent_mse": 1.0,
        }
        for index in range(n)
    ]


def test_post_derangement_is_deterministic_scoped_and_does_not_mutate():
    rows = _rows()
    original = copy.deepcopy(rows)
    first = make_post_diagnostic_rows(rows, "shuffled_pred", 17)
    second = make_post_diagnostic_rows(rows, "shuffled_pred", 17)
    assert [row["post_donor_patient_id"] for row in first] == [
        row["post_donor_patient_id"] for row in second
    ]
    assert all(
        row["patient_id"] != row["post_donor_patient_id"] for row in first
    )
    assert all(torch.equal(row["post_mean"], before["post_mean"]) for row, before in zip(rows, original))
    assert all(first[index]["event"] == rows[index]["event"] for index in range(len(rows)))
    bad = _rows()
    bad[-1]["fold"] = 1
    with pytest.raises(ClarityDiagnosticError, match="cannot be shuffled across"):
        make_post_diagnostic_rows(bad, "shuffled_pred", 17)


def test_observed_post_replaces_only_copied_post():
    rows = _rows()
    changed = make_post_diagnostic_rows(rows, "observed_post")
    for source, row in zip(rows, changed):
        assert torch.equal(row["post_mean"], source["true_post"])
        assert torch.equal(source["post_mean"], torch.tensor([float(source["patient_id"][1:]), 1.0]))
        assert row["post_donor_patient_id"] == source["patient_id"]


def test_checkpoint_audit_distinguishes_valid_epoch0_and_hash_mismatch():
    state = {"weight": torch.tensor([1.0])}
    state_hash = _tensor_state_hash(state)
    snapshot = {
        "total": 1.0,
        "cox_partial_nll": 1.2,
        "bce_identifiable": 0.8,
        "n_events": 2,
        "n_identifiable": 3,
        "risk_variance": 0.1,
        "survival_logit_variance": 0.2,
    }
    training = {
        "variant": "baseline",
        "fold": 0,
        "seed": 7,
        "horizon": 2,
        "available": True,
        "best_epoch": 0,
        "best_stop_loss": 1.0,
        "epochs_ran": 1,
        "fit_n": 4,
        "fit_events": 2,
        "stop_n": 3,
        "stop_events": 1,
        "history": [
            {"epoch": 0, "train": snapshot, "stop": snapshot, "gradient_norm": None},
            {
                "epoch": 1,
                "train": {**snapshot, "total": 0.9},
                "stop": {**snapshot, "total": 1.1},
                "gradient_norm": 0.4,
            },
        ],
    }
    model = {
        "available": True,
        "best_epoch": 0,
        "state": state,
        "init_state_hash": state_hash,
        "final_state_hash": state_hash,
    }
    result = audit_training_task("baseline/fold0/seed7/H2", training, model)
    assert result["checkpoint_consistent"] is True
    assert result["checkpoint_category"] == "initialization"
    assert result["review_reasons"] == [
        "selected_initialization",
        "train_improved_while_last_stop_worsened",
    ]
    assert _audit_summary({"task": result})["review_reason_counts"] == {
        "selected_initialization": 1,
        "train_improved_while_last_stop_worsened": 1,
    }

    broken = {**model, "final_state_hash": "wrong"}
    result = audit_training_task("baseline/fold0/seed7/H2", training, broken)
    assert result["checkpoint_consistent"] is False
    assert "final_state_hash_mismatch" in result["errors"]


def test_diagnostic_config_is_primary_h2_and_read_only_source():
    spec = load_diagnostic_config("configs/v3/clarity_outcome_diagnostics.yaml")
    assert spec["source_run"] == "next_stage_clarity_outcome_v1"
    assert spec["run"] != spec["source_run"]
    assert spec["horizons"] == [2]
    assert spec["permutation_repeats"] == 20

def test_post_summary_keeps_trained_and_initialization_tasks_separate():
    def metric(value):
        return {"value": value}

    def task(category, seed, value):
        return {
            "available": True,
            "variant": "baseline",
            "fold": 0,
            "seed": seed,
            "horizon": 2,
            "checkpoint_category": category,
            "source_reproduction": {"matches": True},
            "normal": {"c_index_risk": metric(value), "ipcw_brier365": metric(value)},
            "shuffled_pred": {"summary": {
                "delta_c_index": {"mean": value},
                "delta_brier": {"mean": value},
                "mean_abs_probability_change": {"mean": value},
                "mean_abs_risk_change": {"mean": value},
            }},
            "observed_post": {
                "metrics": {"c_index_risk": metric(value), "ipcw_brier365": metric(value)},
                "delta_vs_normal": {
                    "c_index_improvement": value,
                    "brier_improvement": value,
                },
            },
        }

    summary = _post_summary({
        "trained": task("trained", 7, 1.0),
        "initial": task("initialization", 17, 2.0),
    })
    split = summary["by_variant_horizon"]["baseline/H2"]["by_checkpoint_category"]
    assert split["trained"]["aggregate"]["normal_c_index"]["mean"] == 1.0
    assert split["initialization"]["aggregate"]["normal_c_index"]["mean"] == 2.0

