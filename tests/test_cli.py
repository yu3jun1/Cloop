from __future__ import annotations

import sys

from cloop.cli import _training_log


def test_training_log_tees_to_terminal_and_appends(tmp_path, capsys):
    config = {"paths": {"project_root": str(tmp_path)}}

    with _training_log(config, "example_run", "train/dynamics") as log_path:
        print("epoch=1 train=0.5")
        print("warning message", file=sys.stderr)
    with _training_log(config, "example_run", "train/outcome"):
        print("epoch=2 train=0.4")

    captured = capsys.readouterr()
    content = log_path.read_text(encoding="utf-8")
    assert "epoch=1 train=0.5" in captured.out
    assert "epoch=2 train=0.4" in captured.out
    assert "warning message" in captured.err
    assert "cloop train/dynamics started" in content
    assert "cloop train/dynamics completed" in content
    assert "cloop train/outcome started" in content
    assert "epoch=1 train=0.5" in content
    assert "epoch=2 train=0.4" in content
    assert "warning message" in content
