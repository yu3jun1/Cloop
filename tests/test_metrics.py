from __future__ import annotations

import pytest

from cloop.metrics import dynamics_paired_improvements, dynamics_summary


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
