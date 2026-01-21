from __future__ import annotations

import os
from pathlib import Path

import torch

from car_image_classification_using_cnn.model import CarClassificationCNN


def export_onnx(
    checkpoint_path: Path,
    onnx_out: Path,
    num_classes: int = 6,
) -> None:
    device = torch.device("cpu")

    model = CarClassificationCNN(num_classes=num_classes, pretrained=False).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)

    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    dummy = torch.randn(1, 3, 224, 224, device=device)

    onnx_out.parent.mkdir(parents=True, exist_ok=True)

    # Classic exporter (works broadly; “dynamic axes” = allow variable batch size) :contentReference[oaicite:4]{index=4}
    torch.onnx.export(
        model,
        (dummy,),
        str(onnx_out),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
        opset_version=17,
    )

    print(f"Exported ONNX to: {onnx_out}")


if __name__ == "__main__":
    # Default follows your project structure
    ckpt = Path(os.getenv("MODEL_PATH", "models/best_model.pth"))
    out = Path("models/best_model.onnx")
    export_onnx(ckpt, out)
