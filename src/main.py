from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from typing import Any
import os

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from torchvision import transforms

from car_image_classification_using_cnn.model import CarClassificationCNN


model: CarClassificationCNN | None = None
device: torch.device
transform: transforms.Compose
class_names: list[str]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, device, transform, class_names

    print("Loading model...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Support local and GCS paths
    model_path_env = os.getenv("MODEL_PATH", "models/best_model.pth")

    if model_path_env.startswith("gs://"):
        from google.cloud import storage

        bucket_name = model_path_env.split("/")[2]
        blob_path = "/".join(model_path_env.split("/")[3:])

        print(f"Downloading model from GCS: {model_path_env}")
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)

        local_model_path = Path("/tmp/best_model.pth")
        blob.download_to_filename(str(local_model_path))
        model_path = local_model_path
        print("Model downloaded from GCS successfully")
    else:
        project_root = Path(__file__).parent.parent
        model_path = project_root / model_path_env

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")

    class_names = ["Audi", "Hyundai Creta", "Rolls Royce", "Swift", "Tata Safari", "Toyota Innova"]

    model = CarClassificationCNN(num_classes=len(class_names), pretrained=False)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    print("Model loaded successfully!")

    yield

    print("Cleaning up...")
    del model


app = FastAPI(
    title="Car Image Classification API",
    description="API for classifying car images using a CNN model trained on 6 car brands",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def read_root() -> dict[str, str]:
    return {
        "message": "Car Image Classification API",
        "status": "active",
        "version": "1.0.0",
        "documentation": "/docs",
    }


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "healthy"}


@app.get("/classes")
def get_classes() -> dict[str, list[str] | int]:
    return {"classes": class_names, "num_classes": len(class_names)}


@app.post("/predict")
async def predict(image: UploadFile = File(...)) -> JSONResponse:
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    try:
        image_data = await image.read()
        img = Image.open(BytesIO(image_data)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

    try:
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)

        predicted_class = class_names[predicted_idx.item()]
        confidence_score = confidence.item()

        all_probabilities = {
            class_name: float(prob) for class_name, prob in zip(class_names, probabilities[0].cpu().numpy())
        }

        return JSONResponse(
            content={
                "predicted_class": predicted_class,
                "confidence": confidence_score,
                "all_probabilities": all_probabilities,
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict_batch")
async def predict_batch(images: list[UploadFile] = File(...)) -> JSONResponse:
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    if len(images) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 images allowed per batch")

    predictions: list[dict[str, Any]] = []

    for idx, image in enumerate(images):
        if not image.content_type or not image.content_type.startswith("image/"):
            predictions.append({"index": idx, "filename": image.filename, "error": "Invalid image file"})
            continue

        try:
            image_data = await image.read()
            img = Image.open(BytesIO(image_data)).convert("RGB")

            img_tensor = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(img_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)

            predicted_class = class_names[predicted_idx.item()]
            confidence_score = confidence.item()

            predictions.append(
                {
                    "index": idx,
                    "filename": image.filename,
                    "predicted_class": predicted_class,
                    "confidence": confidence_score,
                }
            )

        except Exception as e:
            predictions.append({"index": idx, "filename": image.filename, "error": str(e)})

    return JSONResponse(content={"predictions": predictions, "total": len(images)})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
