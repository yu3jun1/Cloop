from __future__ import annotations

import copy
from pathlib import Path

import pytest

from cloop.config import load_config
from cloop.data import fit_preprocessing, load_bundle, make_tiny_cache, split_patients


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config():
    cfg = load_config(ROOT / "configs" / "default.yaml")
    cfg["project"]["device"] = "cpu"
    cfg["data"]["min_action_support"] = 1
    cfg["world"].update(
        hidden_dim=24,
        action_embed_dim=8,
        time_embed_dim=4,
        clinical_embed_dim=6,
        history_embed_dim=6,
        ensemble_size=2,
    )
    cfg["training"].update(seeds=[17], variants=["rrt"], batch_size=8, max_epochs=1, patience=1)
    cfg["outcome"].update(hidden_dim=24, max_epochs=1, patience=1)
    return cfg


@pytest.fixture
def tiny(config):
    cache = make_tiny_cache(num_patients=18, steps=5, latent_dim=8, seed=17)
    split = split_patients(
        [row["patient_id"] for row in cache["patients"]],
        config["data"]["split_seed"],
        config["data"]["train_fraction"],
        config["data"]["validation_fraction"],
    )
    preprocessing = fit_preprocessing(cache, config, split)
    return cache, load_bundle(cache, config, split, preprocessing)

