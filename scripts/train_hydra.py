import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Optional

import hydra
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import webdataset as wds
from omegaconf import OmegaConf
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from validation import validate_data


NUMERIC_FEATURES = ["tenure", "monthly_charges", "support_calls"]
CATEGORICAL_FEATURES = ["contract", "internet_service"]
TARGET_COLUMN = "churn"


def build_sklearn_model(cfg):
    """Create the sklearn preprocessing and classification pipeline from config."""
    hp = cfg.model.hyperparameters

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy=cfg.data.preprocessing.numeric_imputer)),
            ("scaler", StandardScaler()),
        ]
    )

    cat_imputer_cfg = cfg.data.preprocessing.categorical_imputer
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy=cat_imputer_cfg.strategy,
                    fill_value=cat_imputer_cfg.fill_value,
                ),
            ),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )

    classifier_kwargs = {
        "max_iter": hp.max_iter,
        "random_state": hp.random_state,
        "C": hp.C,
        "solver": hp.solver,
        "penalty": hp.penalty,
    }

    return Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            ("classifier", LogisticRegression(**classifier_kwargs)),
        ]
    )


class PyTorchClassifier(nn.Module):
    """Simple PyTorch transformer-based classifier for recognition tasks."""

    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout, vocab_size, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.classifier(x)


class EMA:
    """Exponential Moving Average for model weights."""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            self.shadow[name] = param.data.clone()

    def update(self, model):
        for name, param in model.named_parameters():
            new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
            self.shadow[name] = new_average.clone()

    def apply_shadow(self, model):
        for name, param in model.named_parameters():
            self.backup[name] = param.data.clone()
            param.data = self.shadow[name].clone()

    def restore(self, model):
        for name, param in model.named_parameters():
            param.data = self.backup[name]
        self.backup = {}


def setup_ddp(cfg):
    """Initialize PyTorch DDP for multi-GPU training."""
    if not cfg.training.distributed.use_ddp:
        return False, None

    if not torch.cuda.is_available():
        return False, None

    torch.distributed.init_process_group(
        backend=cfg.training.distributed.backend,
        init_method="env://",
    )
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    return True, torch.distributed


def setup_fsdp(cfg, model):
    """Wrap model with FSDP if enabled."""
    if not cfg.training.distributed.use_fsdp:
        return model

    if not torch.cuda.is_available():
        return model

    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import MixedPrecision

        mp_policy = None
        if cfg.training.distributed.fsdp.mixed_precision == "bf16":
            mp_policy = MixedPrecision(param_dtype=torch.bfloat16)

        model = FSDP(model, mixed_precision=mp_policy)
    except ImportError:
        pass

    return model


def load_webdataset(cfg, split="train"):
    """Create a WebDataset pipeline for streaming NPZ data."""
    data_cfg = cfg.data
    batch_size = data_cfg.batch_size
    num_workers = data_cfg.num_workers

    shard_path = Path(data_cfg.path) / data_cfg.shard_pattern
    shard_url = f"pipe:aws s3 cp {shard_path} - || true" if "s3" in str(data_cfg.path) else str(shard_path).replace("\\", "/")

    dataset = wds.WebDataset(
        shard_url,
        shard_writer=None,
        cache_dir=data_cfg.cache_dir,
    ).shuffle(1000).to_tuple("inputs.npy", "targets.npy").batched(batch_size)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=None,
        num_workers=num_workers,
    )

    return loader


def compute_dataset_hash(filepath):
    """Compute SHA-256 hash of dataset file for versioning."""
    hash_sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()


def setup_mlflow(cfg):
    """Configure MLflow tracking from config."""
    mlflow.set_tracking_uri(cfg.experiment.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.experiment.mlflow.experiment_name)


def log_artifact_with_config(cfg, path, name=None):
    """Log an artifact to MLflow."""
    name = name or Path(path).name
    mlflow.log_artifact(path, artifact_path=name)


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(cfg: OmegaConf) -> None:
    print(f"\nHydra config:\n{OmegaConf.to_yaml(cfg)}")

    setup_mlflow(cfg)

    ckpt_cfg = cfg.training.checkpointing
    ckpt_cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = Path(ckpt_cfg.model_path).parent
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    model_type = cfg.model.type

    if model_type == "sklearn":
        run_sklearn_training(cfg, ckpt_cfg)
    elif model_type == "pytorch":
        run_pytorch_training(cfg, ckpt_cfg)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def run_sklearn_training(cfg, ckpt_cfg):
    """Run sklearn model training with MLflow tracking."""
    df = validate_data(cfg.data.path)

    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = df[features]
    y = df[TARGET_COLUMN].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg.training.test_size,
        random_state=cfg.training.random_state,
        stratify=y,
    )

    model = build_sklearn_model(cfg)

    mlflow_cfg = cfg.experiment.mlflow
    run_name = mlflow_cfg.run_name or f"{cfg.model.name}_{cfg.data.name}"

    with mlflow.start_run(run_name=run_name):
        if mlflow_cfg.log_params:
            hp = cfg.model.hyperparameters
            mlflow.log_params({
                "model_type": hp.model_type,
                "max_iter": hp.max_iter,
                "random_state": hp.random_state,
                "C": hp.C,
                "solver": hp.solver,
                "penalty": hp.penalty,
                "test_size": cfg.training.test_size,
                "minimum_auc": cfg.training.quality_gate.minimum_auc,
            })

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        auc_score = float(roc_auc_score(y_test, probabilities))

        metrics = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision": float(precision_score(y_test, predictions, zero_division=0)),
            "recall": float(recall_score(y_test, predictions, zero_division=0)),
            "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
            "auc": auc_score,
            "roc_auc": auc_score,
            "val_wer": 0.0,
        }

        if mlflow_cfg.log_metrics:
            mlflow.log_metrics(metrics)

        if ckpt_cfg.save_model:
            joblib.dump(model, ckpt_cfg.model_path)

        if ckpt_cfg.save_metrics:
            metrics_path = Path(ckpt_cfg.metrics_path)
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=4)

        if cfg.experiment.dataset_hash.enabled:
            data_hash = compute_dataset_hash(cfg.data.path)
            mlflow.log_param("dataset_sha256", data_hash)

        config_path = Path(hydra.utils.get_config_dir()) / "config.yaml"
        if config_path.exists():
            log_artifact_with_config(cfg, str(config_path), "config")

        if mlflow_cfg.log_artifacts:
            if ckpt_cfg.save_model:
                log_artifact_with_config(cfg, ckpt_cfg.model_path)
            if ckpt_cfg.save_metrics:
                log_artifact_with_config(cfg, ckpt_cfg.metrics_path)

        if mlflow_cfg.log_model:
            mlflow.sklearn.log_model(sk_model=model, artifact_path="model")

        print("\nModel training completed.")
        print(f"Training rows: {len(X_train)}")
        print(f"Testing rows: {len(X_test)}")
        print("\nEvaluation metrics:")
        for name, value in metrics.items():
            print(f"  {name}: {value:.4f}")
        print(f"\nModel saved to: {ckpt_cfg.model_path}")
        print(f"Metrics saved to: {ckpt_cfg.metrics_path}")

        min_auc = cfg.training.quality_gate.minimum_auc
        if auc_score < min_auc:
            raise SystemExit(
                f"QUALITY GATE FAILED: ROC-AUC {auc_score:.4f} "
                f"is below the required {min_auc:.2f}."
            )

        print(f"\nQUALITY GATE PASSED: ROC-AUC {auc_score:.4f} >= {min_auc:.2f}")


def run_pytorch_training(cfg, ckpt_cfg):
    """Run PyTorch model training with DDP/FSDP, WebDataset, EMA, ONNX export."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ddp_enabled, dist = setup_ddp(cfg)

    is_rank_zero = (not ddp_enabled) or (dist.is_initialized and dist.get_rank() == 0)

    hp = cfg.model
    model = PyTorchClassifier(
        d_model=hp.d_model,
        nhead=hp.nhead,
        num_layers=hp.num_layers,
        dim_feedforward=hp.dim_feedforward,
        dropout=hp.dropout,
        vocab_size=hp.vocab_size,
        num_classes=hp.num_classes,
    )
    model = model.to(device)

    if ddp_enabled:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device])

    model = setup_fsdp(cfg, model)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=hp.learning_rate,
        weight_decay=hp.weight_decay,
    )

    ema = None
    if ckpt_cfg.ema.enabled:
        ema = EMA(model, decay=ckpt_cfg.ema.decay)

    train_loader = load_webdataset(cfg, split="train")
    val_loader = load_webdataset(cfg, split="val")

    best_val_wer = np.inf
    max_epochs = cfg.training.max_epochs if hasattr(cfg.training, "max_epochs") else 10

    mlflow_cfg = cfg.experiment.mlflow
    run_name = mlflow_cfg.run_name or f"{cfg.model.name}_{cfg.data.name}"

    with mlflow.start_run(run_name=run_name):
        if mlflow_cfg.log_params:
            mlflow.log_params({
                "model_type": cfg.model.type,
                "arch": cfg.model.arch,
                "d_model": cfg.model.d_model,
                "nhead": cfg.model.nhead,
                "num_layers": cfg.model.num_layers,
                "learning_rate": cfg.model.learning_rate,
                "weight_decay": cfg.model.weight_decay,
                "max_epochs": max_epochs,
                "ddp_enabled": ddp_enabled,
                "fsdp_enabled": cfg.training.distributed.use_fsdp,
            })

        for epoch in range(max_epochs):
            model.train()
            total_loss = 0.0
            num_batches = 0

            for inputs, targets in train_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = nn.functional.cross_entropy(outputs, targets)
                loss.backward()
                optimizer.step()

                if ema:
                    ema.update(model)

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / max(num_batches, 1)
            print(f"Epoch {epoch + 1}/{max_epochs} - Train loss: {avg_loss:.4f}")

            val_metrics = evaluate_pytorch(model, val_loader, device, ema, cfg)

            epoch_metrics = {
                "train_loss": avg_loss,
                "val_wer": val_metrics["wer"],
                "val_accuracy": val_metrics["accuracy"],
            }

            if mlflow_cfg.log_metrics:
                mlflow.log_metrics(epoch_metrics, step=epoch + 1)

            current_wer = val_metrics["wer"]
            if current_wer < best_val_wer:
                best_val_wer = current_wer
                if is_rank_zero and ckpt_cfg.enabled:
                    best_path = ckpt_cfg.checkpoint_dir / "best.pt"
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "epoch": epoch,
                        "val_wer": current_wer,
                        "val_accuracy": val_metrics["accuracy"],
                    }, best_path)

                    if ema and ckpt_cfg.ema.use_ema_for_eval:
                        ema.apply_shadow(model)
                    torch.save(model.state_dict(), ckpt_cfg.checkpoint_dir / "best_ema.pt")
                    if ema:
                        ema.restore(model)

            if cfg.training.checkpointing.onnx.enabled and is_rank_zero:
                try:
                    import torch.onnx
                    dummy_input = torch.randint(0, cfg.model.vocab_size, (1, cfg.model.max_seq_len), device=device)
                    onnx_path = ckpt_cfg.checkpoint_dir / "model.onnx"
                    torch.onnx.export(
                        model,
                        dummy_input,
                        str(onnx_path),
                        export_params=True,
                        opset_version=ckpt_cfg.onnx.opset_version,
                    )
                except Exception as e:
                    print(f"ONNX export failed: {e}")

        final_metrics = {
            "best_val_wer": best_val_wer,
            "final_val_wer": val_metrics["wer"],
            "final_val_accuracy": val_metrics["accuracy"],
        }

        if mlflow_cfg.log_metrics:
            mlflow.log_metrics(final_metrics)

        if mlflow_cfg.log_artifacts and is_rank_zero:
            ckpt_path = ckpt_cfg.checkpoint_dir / "best.pt"
            if ckpt_path.exists():
                mlflow.log_artifact(str(ckpt_path))
            onnx_path = ckpt_cfg.checkpoint_dir / "model.onnx"
            if onnx_path.exists():
                mlflow.log_artifact(str(onnx_path))

        print(f"\nTraining completed. Best val WER: {best_val_wer:.4f}")

        if cfg.training.quality_gate.minimum_wer > 0:
            if best_val_wer > cfg.training.quality_gate.minimum_wer:
                raise SystemExit(
                    f"QUALITY GATE FAILED: WER {best_val_wer:.4f} "
                    f"exceeds maximum {cfg.training.quality_gate.minimum_wer:.2f}."
                )
            print(f"\nQUALITY GATE PASSED: WER {best_val_wer:.4f} <= {cfg.training.quality_gate.minimum_wer:.2f}")


def evaluate_pytorch(model, val_loader, device, ema=None, cfg=None):
    """Evaluate PyTorch model on validation data."""
    model.eval()
    total_correct = 0
    total_samples = 0
    total_edit_distance = 0
    total_chars = 0

    if ema and cfg.training.checkpointing.ema.use_ema_for_eval:
        ema.apply_shadow(model)

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            preds = outputs.argmax(dim=-1)

            total_correct += (preds == targets).sum().item()
            total_samples += targets.size(0)

            pred_str = preds.cpu().numpy()
            target_str = targets.cpu().numpy()
            for p, t in zip(pred_str, target_str):
                total_edit_distance += compute_edit_distance(p.tolist(), t.tolist())
                total_chars += max(len(p), len(t), 1)

    if ema and cfg.training.checkpointing.ema.use_ema_for_eval:
        ema.restore(model)

    accuracy = total_correct / max(total_samples, 1)
    wer = total_edit_distance / max(total_chars, 1)

    return {"accuracy": accuracy, "wer": wer}


def compute_edit_distance(pred, target):
    """Compute Levenshtein edit distance between token sequences."""
    m, n = len(pred), len(target)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred[i - 1] == target[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return dp[m][n]


if __name__ == "__main__":
    main()
