from __future__ import annotations

from pathlib import Path

import bentoml
import numpy as np
from onnxruntime import InferenceSession
from PIL import Image

CLASS_NAMES = ["Audi", "Hyundai Creta", "Rolls Royce", "Swift", "Tata Safari", "Toyota Innova"]


def preprocess_pil(img: Image.Image) -> np.ndarray:
    """
    Match your inference preprocessing:
    - resize 224x224
    - convert to float32
    - CHW
    - normalize like ImageNet
    - add batch dimension
    """
    img = img.convert("RGB").resize((224, 224))
    x = np.asarray(img).astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))  # HWC -> CHW

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
    x = (x - mean) / std

    x = np.expand_dims(x, axis=0)  # (1, 3, 224, 224)
    return x


@bentoml.service(
    resources={"cpu": "2"},
)
class CarClassifierService:
    def __init__(self) -> None:
        model_path = Path("models/best_model.onnx")
        if not model_path.exists():
            raise FileNotFoundError("models/best_model.onnx not found. Run: uv run python scripts/export_onnx.py")

        # CPU inference session
        self.session = InferenceSession(str(model_path))
        self.input_name = self.session.get_inputs()[0].name

    @bentoml.api(
        batchable=True,  # adaptive batching
        batch_dim=(0, 0),  # batch dimension is axis 0 for both input/output
        max_batch_size=32,
        max_latency_ms=200,
    )
    def predict(self, image: np.ndarray) -> list[dict]:
        """
        Input: image as Numpy array (N,3,224,224) float32
        Output: list[dict] length N (one result per item)
        """
        logits = self.session.run(None, {self.input_name: image.astype(np.float32)})[0]

        # softmax over classes
        exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp / np.sum(exp, axis=1, keepdims=True)

        results: list[dict] = []
        for i in range(probs.shape[0]):
            pred_idx = int(np.argmax(probs[i]))
            conf = float(probs[i, pred_idx])

            results.append(
                {
                    "predicted_class": CLASS_NAMES[pred_idx],
                    "confidence": conf,
                    "all_probabilities": {CLASS_NAMES[j]: float(probs[i, j]) for j in range(len(CLASS_NAMES))},
                }
            )

        return results
