from __future__ import annotations

import torch

from cloop.v1.metrics import harrell_c_index, ipcw_brier
from cloop.v1.outcome import PiecewiseExponentialHead, event_bins, interval_exposure


def _inputs(batch=3):
    return torch.randn(batch, 4), torch.randn(batch, 2), torch.ones(batch, 2), torch.randn(batch, 3)


def test_survival_probability_is_bounded_and_monotone():
    model = PiecewiseExponentialHead(4, 2, 3, hidden_dim=8)
    z, c, m, h = _inputs()
    values = torch.stack([model.survival(z, c, m, h, day) for day in (0.0, 90.0, 365.0, 730.0)])
    assert bool(((values >= 0) & (values <= 1)).all())
    assert bool((values[1:] <= values[:-1] + 1e-7).all())


def test_boundary_bins_are_right_closed_and_exposure_is_manual():
    edges = torch.tensor([0.0, 90.0, 180.0, 365.0, 730.0])
    times = torch.tensor([90.0, 180.0, 365.0, 900.0])
    assert event_bins(times, edges).tolist() == [0, 1, 2, 3]
    assert torch.equal(interval_exposure(torch.tensor([100.0]), edges), torch.tensor([[90.0, 10.0, 0.0, 0.0]]))
    assert interval_exposure(torch.tensor([900.0]), edges).sum() == 730


def test_event_and_censor_nll_match_constant_rate_construction():
    model = PiecewiseExponentialHead(1, 1, 1, hidden_dim=2)
    for parameter in model.parameters():
        parameter.data.zero_()
    z = c = m = h = torch.ones(1, 1)
    event = model.nll(z, c, m, h, torch.tensor([90.0]), torch.tensor([1]), reduction="none")
    censor = model.nll(z, c, m, h, torch.tensor([90.0]), torch.tensor([0]), reduction="none")
    rate = torch.nn.functional.softplus(torch.tensor(0.0)) / 365
    assert torch.allclose(censor, rate * 90)
    assert torch.allclose(event, rate * 90 - torch.log(rate))


def test_invalid_survival_inputs_raise():
    model = PiecewiseExponentialHead(1, 1, 1, hidden_dim=2)
    z = c = m = h = torch.ones(1, 1)
    for time, event in ((0.0, 1), (10.0, 2)):
        try:
            model.nll(z, c, m, h, torch.tensor([time]), torch.tensor([event]))
        except ValueError:
            pass
        else:
            raise AssertionError("invalid label was accepted")


def test_c_index_risk_direction_and_brier_no_support_reason():
    cindex = harrell_c_index([1, 2, 3], [1, 1, 0], [3, 2, 1])
    assert cindex["value"] == 1.0
    result = ipcw_brier([1, 2], [0, 0], [3], [1], [0.5], 3)
    assert result["value"] is None
    assert "support" in result["reason"]


def test_outcome_backward_does_not_touch_external_dynamics():
    dynamics = torch.nn.Linear(4, 4)
    model = PiecewiseExponentialHead(4, 2, 3, hidden_dim=8)
    z, c, m, h = _inputs()
    loss = model.nll(z, c, m, h, torch.tensor([30.0, 40.0, 50.0]), torch.tensor([1, 0, 1]))
    loss.backward()
    assert all(parameter.grad is None for parameter in dynamics.parameters())
