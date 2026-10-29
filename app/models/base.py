from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch


@dataclass
class ModelConfig:
    input_dim: int = 543
    hidden_dim: int = 512
    num_layers: int = 6
    num_heads: int = 8
    ffn_dim: int = 2048
    dropout: float = 0.1
    chunk_size: int = 16
    chunk_overlap: int = 8
    max_seq_len: int = 10000
    languages: list[str] = field(
        default_factory=lambda: ["en", "de"]
    )
    vocab_size: int = 1000
    blank_id: int = 0


@dataclass
class InferenceResult:
    logits: torch.Tensor
    predictions: list[str]
    timestamps: list[list[float]]
    language: str
    latency_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)


class ModelAdapter(abc.ABC):
    """Abstract base for continuous sign language recognition models."""

    @abc.abstractmethod
    def load(self, checkpoint_path: str) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def infer(
        self,
        keypoints: np.ndarray | torch.Tensor,
        language: str = "en",
        chunk_size: Optional[int] = None,
        overlap: Optional[int] = None,
    ) -> InferenceResult:
        raise NotImplementedError

    @abc.abstractmethod
    def get_config(self) -> ModelConfig:
        raise NotImplementedError

    @abc.abstractmethod
    def to(self, device: str | torch.device) -> "ModelAdapter":
        raise NotImplementedError
