from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from .base import InferenceResult, ModelAdapter, ModelConfig


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class LandmarkNormalizer(nn.Module):
    """Normalize MediaPipe Holistic keypoints (543-dim)."""

    def __init__(self, input_dim: int = 543):
        super().__init__()
        self.input_dim = input_dim
        self.pose_dim = 33 * 4
        self.face_dim = 468 * 3
        self.left_dim = 21 * 3
        self.right_dim = 21 * 3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class TransformerCTCModel(nn.Module):
    """Transformer encoder + CTC decoder for continuous sign language recognition.

    Architecture:
        543-d input -> Linear(512) -> 6-layer TransformerEncoder
        (8 heads, FFN 2048) -> CTC head per language
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.normalizer = LandmarkNormalizer(config.input_dim)

        self.input_projection = nn.Linear(config.input_dim, config.hidden_dim)
        self.pos_encoder = PositionalEncoding(
            config.hidden_dim, config.max_seq_len, config.dropout
        )

        encoder_layer = TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.ffn_dim,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layer, num_layers=config.num_layers
        )

        self.language_heads: nn.ModuleDict[str, nn.Linear] = nn.ModuleDict(
            {
                lang: nn.Linear(config.hidden_dim, config.vocab_size)
                for lang in config.languages
            }
        )

        self.ctc_loss_fn = nn.CTCLoss(
            blank=config.blank_id,
            zero_infinity=True,
        )

        # Auxiliary isolated loss head for frame-level classification
        self.auxiliary_isolated_head = nn.ModuleDict(
            {
                lang: nn.Linear(config.hidden_dim, config.vocab_size)
                for lang in config.languages
            }
        )
        self.cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=-100)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        language: str = "en",
        return_logits: bool = True,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        x = self.normalizer(x)
        x = self.input_projection(x)
        x = self.pos_encoder(x.transpose(0, 1)).transpose(
            0, 1
        )
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)

        if not return_logits:
            return {"features": x}

        if language in self.language_heads:
            logits = self.language_heads[language](x)
        else:
            logits = self.language_heads[self.config.languages[0]](x)
        return logits

    def compute_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
        input_lengths: torch.Tensor,
    ) -> torch.Tensor:
        log_probs = torch.log_softmax(logits.transpose(0, 1), dim=-1)
        return self.ctc_loss_fn(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
        )

    def compute_auxiliary_isolated_loss(
        self,
        features: torch.Tensor,
        frame_targets: torch.Tensor,
        language: str = "en",
    ) -> torch.Tensor:
        """Frame-level auxiliary loss for isolated sign classification."""
        if language not in self.auxiliary_isolated_head:
            language = self.config.languages[0]
        head = self.auxiliary_isolated_head[language]
        predictions = head(features)
        # predictions: (batch, seq_len, vocab_size)
        # frame_targets: (batch, seq_len)
        predictions_flat = predictions.reshape(-1, predictions.size(-1))
        targets_flat = frame_targets.reshape(-1)
        mask = targets_flat >= 0
        if mask.sum() == 0:
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)
        return self.cross_entropy_loss(
            predictions_flat[mask], targets_flat[mask]
        )


class ContinuousAdapter(ModelAdapter):
    """ModelAdapter implementation for continuous sign language recognition.

    Uses Transformer encoder + CTC decoder with multi-language support.
    Supports streaming chunked inference for low-latency operation.
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.model: Optional[TransformerCTCModel] = None
        self.device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._vocab: dict[str, list[str]] = {}
        self._initialized = False

    def load(self, checkpoint_path: str) -> None:
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            self.model = TransformerCTCModel(self.config).to(self.device)
            config_path = ckpt_path.parent / "config.json"
            if config_path.exists():
                with open(config_path) as f:
                    saved_config = json.load(f)
                    self.config = ModelConfig(**saved_config)
            self._initialized = True
            return

        checkpoint = torch.load(
            str(ckpt_path),
            map_location=self.device,
            weights_only=False,
        )

        if "config" in checkpoint:
            self.config = ModelConfig(**checkpoint["config"])

        self.model = TransformerCTCModel(self.config).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self._initialized = True

        if "vocab" in checkpoint:
            self._vocab = checkpoint["vocab"]

    def get_config(self) -> ModelConfig:
        return self.config

    def to(self, device: str | torch.device) -> "ContinuousAdapter":
        self.device = torch.device(device)
        if self.model is not None:
            self.model = self.model.to(self.device)
        return self

    def infer(
        self,
        keypoints: np.ndarray | torch.Tensor,
        language: str = "en",
        chunk_size: Optional[int] = None,
        overlap: Optional[int] = None,
    ) -> InferenceResult:
        start_time = time.perf_counter()

        if not self._initialized or self.model is None:
            self.model = TransformerCTCModel(self.config).to(self.device)
            self._initialized = True

        self.model.eval()

        chunk_size = chunk_size or self.config.chunk_size
        overlap = overlap or self.config.chunk_overlap

        if isinstance(keypoints, np.ndarray):
            keypoints = torch.from_numpy(keypoints).float()
        elif keypoints.dtype != torch.float32:
            keypoints = keypoints.float()

        keypoints = keypoints.to(self.device)
        if keypoints.dim() == 2:
            keypoints = keypoints.unsqueeze(0)
        keypoints = keypoints

        with torch.no_grad():
            seq_len = keypoints.size(1)
            use_streaming = seq_len > chunk_size * 2

            if use_streaming:
                logits = self._streaming_infer(
                    keypoints, language, chunk_size, overlap
                )
            else:
                logits = self.model(keypoints, language=language)

            log_probs = torch.log_softmax(
                logits.transpose(0, 1), dim=-1
            )
            predictions_list = self._ctc_decode(
                log_probs, language
            )

        timestamps = self._generate_timestamps(
            log_probs.shape[0], predictions_list
        )

        latency_ms = (time.perf_counter() - start_time) * 1000

        return InferenceResult(
            logits=logits,
            predictions=predictions_list,
            timestamps=timestamps,
            language=language,
            latency_ms=latency_ms,
            metadata={
                "seq_len": seq_len,
                "chunk_size": chunk_size,
                "overlap": overlap,
                "streaming": use_streaming,
            },
        )

    def _streaming_infer(
        self,
        keypoints: torch.Tensor,
        language: str,
        chunk_size: int,
        overlap: int,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = keypoints.shape
        outputs = []
        step = chunk_size - overlap

        for i in range(0, seq_len, step):
            start = i
            end = min(i + chunk_size, seq_len)
            chunk = keypoints[:, start:end, :]

            chunk_logits = self.model(
                chunk, language=language
            )
            outputs.append(chunk_logits)

        concatenated = torch.cat(outputs, dim=1)
        # Trim overlap regions from concatenated output
        if len(outputs) > 1 and overlap > 0:
            trimmed_outputs = [outputs[0]]
            for out in outputs[1:]:
                trimmed_outputs.append(out[:, overlap:, :])
            concatenated = torch.cat(trimmed_outputs, dim=1)
        return concatenated

    def _ctc_decode(
        self,
        log_probs: torch.Tensor,
        language: str,
    ) -> list[str]:
        predictions = []
        batch_size = log_probs.size(1)

        for b in range(batch_size):
            argmax_indices = log_probs.argmax(dim=-1)[:, b]
            collapsed = []
            prev = -1
            for idx in argmax_indices:
                if idx != prev and idx != self.config.blank_id:
                    collapsed.append(idx.item())
                prev = idx
            tokens = collapsed if collapsed else [0]
            text = "".join(f"<tok_{t}>" for t in tokens)
            predictions.append(text)

        return predictions

    def _generate_timestamps(
        self,
        seq_len: int,
        predictions: list[str],
    ) -> list[list[float]]:
        return [
            [float(i) * 1000.0 / max(seq_len, 1) for i in range(seq_len)]
            for p in predictions
        ]
