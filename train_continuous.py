"""Training script: CTC loss + auxiliary isolated loss."""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW

from app.models.continuous_adapter import (
    ContinuousAdapter,
    ModelConfig,
    TransformerCTCModel,
)


def train_step(
    model: TransformerCTCModel,
    optimizer: torch.optim.Optimizer,
    keypoints: torch.Tensor,
    targets: torch.Tensor,
    target_lengths: torch.Tensor,
    input_lengths: torch.Tensor,
    frame_targets: torch.Tensor,
    language: str = "en",
    ctc_weight: float = 0.8,
    aux_weight: float = 0.2,
) -> dict:
    optimizer.zero_grad()
    logits = model(keypoints, language=language)
    ctc_loss = model.compute_loss(
        logits,
        targets,
        target_lengths,
        input_lengths,
    )
    # Auxiliary isolated loss requires features (before output head)
    # Get encoder features by using return_logits=False and then project
    with torch.no_grad():
        features = model(keypoints, language=language, return_logits=False)["features"]
    # Re-enable gradients for feature extraction in auxiliary path
    features = model.normalizer(keypoints)
    features = model.input_projection(features)
    features = model.pos_encoder(features.transpose(0, 1)).transpose(0, 1)
    features = model.transformer_encoder(features)

    aux_loss = model.compute_auxiliary_isolated_loss(
        features, frame_targets, language=language
    )
    total_loss = ctc_weight * ctc_loss + aux_weight * aux_loss
    total_loss.backward()
    optimizer.step()

    return {
        "total_loss": total_loss.item(),
        "ctc_loss": ctc_loss.item(),
        "aux_loss": aux_loss.item(),
    }


def build_training_dataset():
    # Placeholder: in production load PHOENIX-2014T / ASL-Citizen
    return None


def main():
    parser = argparse.ArgumentParser(description="Train continuous SLR model")
    parser.add_argument(
        "--checkpoints", default="artifacts/model.pt", help="Checkpoint path"
    )
    args = parser.parse_args()
    adapter = ContinuousAdapter(config=ModelConfig())
    adapter.load(args.checkpoints)
    print(f"Loaded adapter. Model initialized: {adapter._initialized}")
    print(f"Config: hidden={adapter.config.hidden_dim}, layers={adapter.config.num_layers}")


if __name__ == "__main__":
    main()
