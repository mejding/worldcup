import json
from pathlib import Path

from config import MODEL_METADATA_PATH, MODEL_PATH


def model_exists(model_path: Path = MODEL_PATH) -> bool:
    return Path(model_path).exists()


def load_model_metadata(path: Path = MODEL_METADATA_PATH) -> dict:
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def get_active_model_status() -> dict:
    metadata = load_model_metadata()
    metrics = metadata.get("metrics", {})
    return {
        "model_exists": model_exists(),
        "trained_at": metadata.get("trained_at"),
        "number_of_training_rows": metadata.get("number_of_training_rows", 0),
        "number_of_test_rows": metadata.get("number_of_test_rows", 0),
        "accuracy": metrics.get("accuracy"),
        "log_loss": metrics.get("log_loss"),
        "brier_score": metrics.get("brier_score"),
        "draw_rate_actual": metrics.get("draw_rate_actual"),
        "draw_rate_predicted": metrics.get("draw_rate_predicted"),
    }

