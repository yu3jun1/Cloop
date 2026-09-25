from __future__ import annotations

import pytest

from cloop.v1.metrics import (
    dynamics_paired_improvements,
    dynamics_summary,
    replanning_behavior_summary,
)


def test_dynamics_summary_names_cosine_similarity_unambiguously():
    summary = dynamics_summary(
        [
            {
                "horizon": 1,
                "mse": 0.2,
                "cosine_similarity": 0.75,
                "patient_id": "P1",
            },
            {
                "horizon": 1,
                "mse": 0.4,
                "cosine_similarity": 0.25,
                "patient_id": "P2",
            },
        ]
    )
    assert summary["CosSim@1"] == pytest.approx(0.5)
    assert "cosine@1" not in summary


def test_dynamics_paired_improvements_include_d1_d2_and_d3_d4():
    result = dynamics_paired_improvements(
        {
            "baseline": {"7": 10.0, "17": 20.0},
            "rrt": {"7": 8.0, "17": 10.0},
            "ensemble": {"7": 8.0, "17": 16.0},
            "rrt_ensemble": {"7": 4.0, "17": 8.0},
        }
    )
    assert set(result) == {
        "baseline_vs_rrt",
        "ensemble_vs_rrt_ensemble",
        "baseline_vs_rrt_ensemble",
    }
    assert result["baseline_vs_rrt"]["mean"] == pytest.approx(35.0)
    assert result["ensemble_vs_rrt_ensemble"]["mean"] == pytest.approx(50.0)
    assert result["baseline_vs_rrt_ensemble"]["mean"] == pytest.approx(60.0)


def test_plan_revision_differs_from_sequential_action_change():
    rows = [
        {
            "patient_id": "P001",
            "recommended_action": "A",
            "plan_revision_evaluable": False,
            "plan_revised": None,
        },
        {
            "patient_id": "P001",
            "recommended_action": "B",
            "plan_revision_evaluable": True,
            "plan_revised": False,
        },
    ]

    summary = replanning_behavior_summary(rows, "mpc_rrt_ensemble")

    assert summary["sequential_action_change_rate"] == pytest.approx(1.0)
    assert summary["plan_revision_rate"] == pytest.approx(0.0)
    assert summary["plan_revision_evaluable_count"] == 1


def test_plan_revision_reasons_for_non_replanning_methods():
    rows = [
        {
            "patient_id": "P001",
            "recommended_action": "A",
            "plan_revision_evaluable": False,
            "plan_revised": None,
        },
        {
            "patient_id": "P001",
            "recommended_action": "D",
            "plan_revision_evaluable": True,
            "plan_revised": True,
        },
    ]

    mpc = replanning_behavior_summary(rows, "mpc_rrt_ensemble")
    assert mpc["plan_revision_rate"] == pytest.approx(1.0)

    greedy = replanning_behavior_summary(rows, "greedy")
    assert greedy["plan_revision_rate"] is None
    assert greedy["plan_revision_reason"] == "no_prior_future_action_in_h1_policy"

    fixed = replanning_behavior_summary(rows, "fixed_plan")
    assert fixed["plan_revision_rate"] is None
    assert fixed["plan_revision_reason"] == "fixed_plan_does_not_replan"
