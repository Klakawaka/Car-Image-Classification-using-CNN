import pytest
import torch
from pathlib import Path

from car_image_classification_using_cnn.model import CarClassificationCNN


@pytest.fixture(scope="session")
def temp_model_file(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create a temporary model checkpoint compatible with the API startup."""
    tmp_dir = tmp_path_factory.mktemp("api_model")
    ckpt_path = tmp_dir / "best_model.pth"

    # Match src/main.py class_names length (6)
    model = CarClassificationCNN(num_classes=6, pretrained=False)
    torch.save({"model_state_dict": model.state_dict()}, ckpt_path)
    return ckpt_path


@pytest.fixture
def api_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, temp_model_file: Path) -> None:
    """Prepare env + working directory so API endpoints can run in isolation."""
    monkeypatch.setenv("MODEL_PATH", str(temp_model_file))
    # Avoid writing prediction_database.csv into repo root during tests
    monkeypatch.chdir(tmp_path)
