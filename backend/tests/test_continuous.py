from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from app.models.base import InferenceResult, ModelAdapter, ModelConfig
from app.models.continuous_adapter import (
    ContinuousAdapter,
    PositionalEncoding,
    TransformerCTCModel,
)


@pytest.fixture
def model_config():
    return ModelConfig(
        input_dim=543,
        hidden_dim=128,
        num_layers=2,
        num_heads=4,
        ffn_dim=512,
        dropout=0.1,
        chunk_size=16,
        chunk_overlap=8,
        max_seq_len=1000,
        languages=["en", "de"],
        vocab_size=100,
        blank_id=0,
    )


@pytest.fixture
def adapter(model_config):
    return ContinuousAdapter(config=model_config)


@pytest.fixture
def keypoints(batch_size=1, seq_len=32, dim=543):
    return np.random.randn(batch_size, seq_len, dim).astype(np.float32)


class TestModelConfig:
    def test_default_config(self):
        config = ModelConfig()
        assert config.input_dim == 543
        assert config.hidden_dim == 512
        assert config.num_layers == 6
        assert config.num_heads == 8
        assert config.ffn_dim == 2048
        assert config.chunk_size == 16
        assert config.chunk_overlap == 8
        assert "en" in config.languages
        assert "de" in config.languages

    def test_config_with_custom_values(self):
        config = ModelConfig(
            input_dim=256,
            hidden_dim=64,
            num_layers=1,
            num_heads=2,
            ffn_dim=128,
        )
        assert config.input_dim == 256
        assert config.hidden_dim == 64
        assert config.num_layers == 1


class TestPositionalEncoding:
    def test_forward_shape(self, model_config):
        pe = PositionalEncoding(model_config.hidden_dim, max_len=100)
        x = torch.randn(10, 2, model_config.hidden_dim)
        out = pe(x)
        assert out.shape == x.shape

    def test_forward_different_lengths(self, model_config):
        pe = PositionalEncoding(model_config.hidden_dim, max_len=1000)
        x1 = torch.randn(50, 1, model_config.hidden_dim)
        x2 = torch.randn(100, 1, model_config.hidden_dim)
        assert pe(x1).shape == x1.shape
        assert pe(x2).shape == x2.shape

    def test_dropout_in_eval_mode(self, model_config):
        pe = PositionalEncoding(model_config.hidden_dim, max_len=100)
        pe.eval()
        x = torch.randn(10, 2, model_config.hidden_dim)
        out1 = pe(x)
        out2 = pe(x)
        assert torch.allclose(out1, out2)


class TestTransformerCTCModel:
    def test_forward_output_shape(self, model_config):
        model = TransformerCTCModel(model_config)
        x = torch.randn(2, 32, model_config.input_dim)
        logits = model(x, language="en")
        assert logits.shape == (2, 32, model_config.vocab_size)

    def test_multi_language_heads(self, model_config):
        model = TransformerCTCModel(model_config)
        assert "en" in model.language_heads
        assert "de" in model.language_heads
        x = torch.randn(1, 16, model_config.input_dim)
        en_logits = model(x, language="en")
        de_logits = model(x, language="de")
        assert en_logits.shape == (1, 16, model_config.vocab_size)
        assert de_logits.shape == (1, 16, model_config.vocab_size)
        assert not torch.allclose(en_logits, de_logits)

    def test_return_features(self, model_config):
        model = TransformerCTCModel(model_config)
        x = torch.randn(1, 16, model_config.input_dim)
        result = model(x, return_logits=False)
        assert "features" in result
        assert result["features"].shape == (1, 16, model_config.hidden_dim)

    def test_padding_mask(self, model_config):
        model = TransformerCTCModel(model_config)
        x = torch.randn(1, 32, model_config.input_dim)
        padding_mask = torch.zeros(1, 32, dtype=torch.bool)
        padding_mask[0, 20:] = True
        logits = model(x, src_key_padding_mask=padding_mask, language="en")
        assert logits.shape == (1, 32, model_config.vocab_size)

    def test_ctc_loss(self, model_config):
        model = TransformerCTCModel(model_config)
        logits = torch.randn(16, 2, model_config.vocab_size)
        targets = torch.tensor([[1, 2, 3], [4, 5, 0]])
        target_lengths = torch.tensor([3, 2])
        input_lengths = torch.tensor([16, 16])
        loss = model.compute_loss(logits, targets, target_lengths, input_lengths)
        assert loss.item() > 0
        assert loss.requires_grad

    def test_default_language_fallback(self, model_config):
        model = TransformerCTCModel(model_config)
        x = torch.randn(1, 8, model_config.input_dim)
        logits = model(x, language="nonexistent")
        assert logits.shape == (1, 8, model_config.vocab_size)

    def test_weight_initialization(self, model_config):
        model = TransformerCTCModel(model_config)
        for name, param in model.named_parameters():
            if param.dim() > 1:
                assert param.abs().mean() < 2.0


class TestContinuousAdapter:
    def test_adapter_is_model_adapter(self, adapter):
        assert isinstance(adapter, ModelAdapter)

    def test_get_config(self, adapter, model_config):
        config = adapter.get_config()
        assert config.input_dim == model_config.input_dim
        assert config.num_layers == model_config.num_layers

    def test_load_nonexistent_checkpoint(self, adapter, tmp_path):
        checkpoint_path = str(tmp_path / "nonexistent.pt")
        adapter.load(checkpoint_path)
        assert adapter.model is not None
        assert adapter._initialized is True

    def test_load_with_checkpoint(self, adapter, tmp_path, model_config):
        model = TransformerCTCModel(model_config)
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "config": {
                "input_dim": model_config.input_dim,
                "hidden_dim": model_config.hidden_dim,
                "num_layers": model_config.num_layers,
                "num_heads": model_config.num_heads,
                "ffn_dim": model_config.ffn_dim,
                "dropout": model_config.dropout,
                "chunk_size": model_config.chunk_size,
                "chunk_overlap": model_config.chunk_overlap,
                "max_seq_len": model_config.max_seq_len,
                "languages": model_config.languages,
                "vocab_size": model_config.vocab_size,
                "blank_id": model_config.blank_id,
            },
        }
        checkpoint_path = tmp_path / "model.pt"
        torch.save(checkpoint, str(checkpoint_path))
        adapter.load(str(checkpoint_path))
        assert adapter.model is not None

    def test_to_device(self, adapter):
        device = torch.device("cpu")
        result = adapter.to(device)
        assert isinstance(result, ContinuousAdapter)
        assert adapter.device == device

    def test_infer_basic_keypoints(self, adapter, keypoints):
        result = adapter.infer(keypoints, language="en")
        assert isinstance(result, InferenceResult)
        assert result.logits is not None
        assert len(result.predictions) == 1
        assert result.language == "en"
        assert result.latency_ms > 0

    def test_infer_output_shape(self, adapter, keypoints):
        result = adapter.infer(keypoints, language="en")
        expected_seq_len = keypoints.shape[1]
        assert result.logits.shape == (
            1,
            expected_seq_len,
            adapter.config.vocab_size,
        )

    def test_infer_ctc_output_format(self, adapter, keypoints):
        result = adapter.infer(keypoints, language="en")
        logits = result.logits
        log_probs = torch.log_softmax(
            logits.transpose(0, 1), dim=-1
        )
        assert log_probs.shape[0] == logits.shape[1]
        assert log_probs.shape[1] == logits.shape[0]
        assert log_probs.shape[2] == adapter.config.vocab_size
        assert torch.allclose(
            log_probs.exp().sum(dim=-1),
            torch.ones_like(log_probs.exp().sum(dim=-1)),
            atol=1e-5,
        )

    def test_infer_with_torch_tensor(self, adapter):
        keypoints = torch.randn(1, 32, 543)
        result = adapter.infer(keypoints, language="en")
        assert isinstance(result, InferenceResult)
        assert len(result.predictions) == 1

    def test_infer_with_2d_input(self, adapter):
        keypoints = torch.randn(32, 543)
        result = adapter.infer(keypoints, language="en")
        assert isinstance(result, InferenceResult)
        assert result.logits.shape[0] == 1

    def test_infer_batch(self, adapter):
        keypoints = np.random.randn(4, 32, 543).astype(np.float32)
        result = adapter.infer(keypoints, language="en")
        assert len(result.predictions) == 4
        assert result.logits.shape[0] == 4

    def test_infer_streaming(self, adapter):
        seq_len = 100
        keypoints = np.random.randn(1, seq_len, 543).astype(np.float32)
        result = adapter.infer(
            keypoints,
            language="en",
            chunk_size=16,
            overlap=8,
        )
        assert result.metadata["streaming"] is True
        assert result.logits.shape[1] == seq_len

    def test_infer_non_streaming(self, adapter):
        seq_len = 20
        keypoints = np.random.randn(1, seq_len, 543).astype(np.float32)
        result = adapter.infer(
            keypoints,
            language="en",
            chunk_size=16,
            overlap=8,
        )
        assert result.metadata["streaming"] is False

    def test_infer_multi_language(self, adapter, keypoints):
        result_en = adapter.infer(keypoints, language="en")
        result_de = adapter.infer(keypoints, language="de")
        assert result_en.language == "en"
        assert result_de.language == "de"
        assert not torch.allclose(
            result_en.logits, result_de.logits
        )

    def test_infer_timestamps(self, adapter, keypoints):
        result = adapter.infer(keypoints, language="en")
        assert len(result.timestamps) == len(result.predictions)
        for ts in result.timestamps:
            assert len(ts) <= result.logits.shape[1]

    def test_infer_metadata(self, adapter, keypoints):
        result = adapter.infer(keypoints, language="en")
        assert "seq_len" in result.metadata
        assert "chunk_size" in result.metadata
        assert "overlap" in result.metadata
        assert "streaming" in result.metadata

    def test_infer_latency_within_target(self, adapter):
        seq_len = 32
        keypoints = np.random.randn(1, seq_len, 543).astype(np.float32)
        result = adapter.infer(keypoints, language="en")
        assert result.latency_ms < 200

    def test_infer_predictions_are_strings(self, adapter, keypoints):
        result = adapter.infer(keypoints, language="en")
        for pred in result.predictions:
            assert isinstance(pred, str)


class TestCTCDecoding:
    def test_ctc_decode_collapses_repeats(self, model_config):
        model = TransformerCTCModel(model_config)
        log_probs = torch.randn(10, 1, model_config.vocab_size)
        log_probs = log_probs.softmax(dim=-1)
        predictions = ContinuousAdapter._ctc_decode(
            ContinuousAdapter(), log_probs, "en"
        )
        assert isinstance(predictions, list)


class TestStreamingInference:
    def test_streaming_output_length(self, model_config):
        adapter = ContinuousAdapter(config=model_config)
        seq_len = 64
        chunk_size = 16
        overlap = 8
        keypoints = np.random.randn(1, seq_len, 543).astype(np.float32)
        result = adapter.infer(
            keypoints,
            language="en",
            chunk_size=chunk_size,
            overlap=overlap,
        )
        assert result.logits.shape[1] == seq_len

    def test_different_chunk_sizes(self, model_config):
        adapter = ContinuousAdapter(config=model_config)
        keypoints = np.random.randn(1, 50, 543).astype(np.float32)
        result_a = adapter.infer(
            keypoints, language="en", chunk_size=16, overlap=8
        )
        result_b = adapter.infer(
            keypoints, language="en", chunk_size=8, overlap=4
        )
        assert result_a.logits.shape == result_b.logits.shape


class TestModelPersistence:
    def test_save_and_load(self, adapter, model_config, tmp_path):
        model = TransformerCTCModel(model_config)
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "config": {
                "input_dim": model_config.input_dim,
                "hidden_dim": model_config.hidden_dim,
                "num_layers": model_config.num_layers,
                "num_heads": model_config.num_heads,
                "ffn_dim": model_config.ffn_dim,
                "dropout": model_config.dropout,
                "chunk_size": model_config.chunk_size,
                "chunk_overlap": model_config.chunk_overlap,
                "max_seq_len": model_config.max_seq_len,
                "languages": model_config.languages,
                "vocab_size": model_config.vocab_size,
                "blank_id": model_config.blank_id,
            },
        }
        checkpoint_path = tmp_path / "model.pt"
        torch.save(checkpoint, str(checkpoint_path))

        new_adapter = ContinuousAdapter(config=model_config)
        new_adapter.load(str(checkpoint_path))
        assert new_adapter.model is not None
        assert new_adapter._initialized
