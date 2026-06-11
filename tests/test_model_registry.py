import json

from model_registry import get_active_model_status, load_model_metadata


def test_load_model_metadata_returns_empty_dict_when_missing(tmp_path):
    assert load_model_metadata(tmp_path / "missing.json") == {}


def test_active_model_status_exposes_draw_context_flag(tmp_path, monkeypatch):
    metadata_path = tmp_path / "model_metadata.json"
    model_path = tmp_path / "model.pkl"
    model_path.write_bytes(b"placeholder")
    metadata_path.write_text(
        json.dumps(
            {
                "include_draw_context_features": True,
                "metrics": {"accuracy": 0.5},
                "number_of_training_rows": 30,
            }
        )
    )
    monkeypatch.setattr("model_registry.MODEL_METADATA_PATH", metadata_path)
    monkeypatch.setattr("model_registry.MODEL_PATH", model_path)

    status = get_active_model_status()

    assert status["model_exists"] is True
    assert status["include_draw_context_features"] is True
    assert status["accuracy"] == 0.5
