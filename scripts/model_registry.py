"""Model registry for versioned artifacts with metadata."""
import hashlib
import json
import sys
from pathlib import Path

import mlflow


def register_model(
    artifact_dir: Path,
    dataset_path: Path,
    config_path: Path,
    metrics_path: Path,
    registry_uri: str = None,
    model_version: int = 1,
):
    """Register a model version with dataset hash, config, metrics."""
    registry_dir = Path(registry_uri or ".models") / f"v{model_version}"
    registry_dir.mkdir(parents=True, exist_ok=True)

    dataset_hash = compute_file_hash(dataset_path)

    model_path = artifact_dir / "model.joblib"
    metrics_data = json.load(open(metrics_path, "r", encoding="utf-8")) if metrics_path.exists() else {}

    metadata = {
        "dataset_path": str(dataset_path),
        "dataset_sha256": dataset_hash,
        "model_version": model_version,
        "metrics": metrics_data,
        "config_path": str(config_path) if config_path.exists() else None,
    }

    metadata_path = registry_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    if model_path.exists():
        import shutil
        shutil.copy(str(model_path), registry_dir / "model.joblib")
    if metrics_path.exists():
        import shutil
        shutil.copy(str(metrics_path), registry_dir / "metrics.json")

    print(f"Registered model v{model_version} at {registry_dir}")
    print(f"Dataset hash: {dataset_hash}")
    print(f"Metrics: {metrics_data}")


def compute_file_hash(filepath: Path):
    hash_sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Register versioned model artifacts")
    parser.add_argument("--artifact-dir", type=str, default="artifacts")
    parser.add_argument("--dataset", type=str, default="data/customers.csv")
    parser.add_argument("--metrics", type=str, default="artifacts/metrics.json")
    parser.add_argument("--config", type=str, default="conf/config.yaml")
    parser.add_argument("--version", type=int, default=1)
    args = parser.parse_args()

    register_model(
        Path(args.artifact_dir),
        Path(args.dataset),
        Path(args.config),
        Path(args.metrics),
        registry_uri=f"models/v{args.version}",
        model_version=args.version,
    )
