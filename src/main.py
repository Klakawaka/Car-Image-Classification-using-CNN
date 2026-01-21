from contextlib import asynccontextmanager
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any
import os
import csv
import pandas as pd
import numpy as np

import torch
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, HTMLResponse
from PIL import Image
from torchvision import transforms
from prometheus_client import CollectorRegistry, Counter, Histogram, Summary, make_asgi_app

from car_image_classification_using_cnn.model import CarClassificationCNN


model: CarClassificationCNN | None = None
device: torch.device
transform: transforms.Compose
class_names: list[str]

PREDICTION_DB = Path("prediction_database.csv")

# Prometheus metrics stuff
METRICS_REGISTRY = CollectorRegistry()

request_counter = Counter("prediction_requests_total", "Total number of prediction requests", registry=METRICS_REGISTRY)

batch_request_counter = Counter(
    "batch_prediction_requests_total", "Total number of batch prediction requests", registry=METRICS_REGISTRY
)

error_counter = Counter("prediction_errors_total", "Total number of prediction errors", registry=METRICS_REGISTRY)

request_latency = Histogram(
    "prediction_request_duration_seconds", "Time spent processing prediction request", registry=METRICS_REGISTRY
)

batch_latency = Histogram(
    "batch_prediction_request_duration_seconds",
    "Time spent processing batch prediction request",
    registry=METRICS_REGISTRY,
)

confidence_summary = Summary(
    "prediction_confidence", "Distribution of prediction confidence scores", registry=METRICS_REGISTRY
)

image_brightness_summary = Summary(
    "image_brightness", "Distribution of image brightness values", registry=METRICS_REGISTRY
)

image_contrast_summary = Summary("image_contrast", "Distribution of image contrast values", registry=METRICS_REGISTRY)


def save_prediction_to_db(
    predicted_class: str,
    confidence: float,
    mean_brightness: float,
    std_brightness: float,
    contrast: float,
    max_brightness: float = 0.0,
    min_brightness: float = 0.0,
) -> None:
    """Save prediction to database as background task."""
    file_exists = PREDICTION_DB.exists()

    with open(PREDICTION_DB, "a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "time", "predicted_class", "confidence", 
                "mean_brightness", "std_brightness", "contrast",
                "max_brightness", "min_brightness"
            ])

        writer.writerow([
            datetime.now().isoformat(), predicted_class, confidence, 
            mean_brightness, std_brightness, contrast,
            max_brightness, min_brightness
        ])


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, device, transform, class_names

    print("Loading model...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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

# Mount prometheus metrics endpoint with custom registry
app.mount("/metrics", make_asgi_app(registry=METRICS_REGISTRY))


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
async def predict(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
) -> JSONResponse:
    request_counter.inc()

    if model is None:
        error_counter.inc()
        raise HTTPException(status_code=500, detail="Model not loaded")

    if not image.content_type or not image.content_type.startswith("image/"):
        error_counter.inc()
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    try:
        image_data = await image.read()
        img = Image.open(BytesIO(image_data)).convert("RGB")
    except Exception as e:
        error_counter.inc()
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

    try:
        img_array = np.array(img, dtype=np.float32)
        mean_brightness = float(img_array.mean())
        std_brightness = float(img_array.std())
        contrast = float(img_array.max() - img_array.min())
        max_brightness = float(img_array.max())
        min_brightness = float(img_array.min())

        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)

        predicted_class_index = int(predicted_idx.item())
        predicted_class = class_names[predicted_class_index]
        confidence_score = float(confidence.item())

        all_probabilities = {
            class_name: float(prob) for class_name, prob in zip(class_names, probabilities[0].cpu().numpy())
        }

        background_tasks.add_task(
            save_prediction_to_db,
            predicted_class,
            confidence_score,
            mean_brightness,
            std_brightness,
            contrast,
            max_brightness,
            min_brightness,
        )
    with request_latency.time():
        try:
            img_array = np.array(img, dtype=np.float32)
            mean_brightness = float(img_array.mean())
            std_brightness = float(img_array.std())
            contrast = float(img_array.max() - img_array.min())

            # Track image metrics
            image_brightness_summary.observe(mean_brightness)
            image_contrast_summary.observe(contrast)

            img_tensor = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(img_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)

            predicted_class_index = int(predicted_idx.item())
            predicted_class = class_names[predicted_class_index]
            confidence_score = float(confidence.item())

            # Track confidence metric
            confidence_summary.observe(confidence_score)

            all_probabilities = {
                class_name: float(prob) for class_name, prob in zip(class_names, probabilities[0].cpu().numpy())
            }

            background_tasks.add_task(
                save_prediction_to_db,
                predicted_class,
                confidence_score,
                mean_brightness,
                std_brightness,
                contrast,
            )

            return JSONResponse(
                content={
                    "predicted_class": predicted_class,
                    "confidence": confidence_score,
                    "all_probabilities": all_probabilities,
                }
            )

        except Exception as e:
            error_counter.inc()
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict_batch")
async def predict_batch(images: list[UploadFile] = File(...)) -> JSONResponse:
    batch_request_counter.inc()

    if model is None:
        error_counter.inc()
        raise HTTPException(status_code=500, detail="Model not loaded")

    if len(images) > 10:
        error_counter.inc()
        raise HTTPException(status_code=400, detail="Maximum 10 images allowed per batch")

    with batch_latency.time():
        predictions: list[dict[str, Any]] = []

        for idx, image in enumerate(images):
            if not image.content_type or not image.content_type.startswith("image/"):
                predictions.append({"index": idx, "filename": image.filename, "error": "Invalid image file"})
                error_counter.inc()
                continue

            try:
                image_data = await image.read()
                img = Image.open(BytesIO(image_data)).convert("RGB")

                img_tensor = transform(img).unsqueeze(0).to(device)

                with torch.no_grad():
                    outputs = model(img_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    confidence, predicted_idx = torch.max(probabilities, 1)

                predicted_class_index = int(predicted_idx.item())
                predicted_class = class_names[predicted_class_index]
                confidence_score = float(confidence.item())

                # Track confidence metric for batch predictions too
                confidence_summary.observe(confidence_score)

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
                error_counter.inc()

        return JSONResponse(content={"predictions": predictions, "total": len(images)})


@app.get("/monitoring", response_class=HTMLResponse)
async def get_monitoring_report(n: int = 100):
    try:
        if not PREDICTION_DB.exists():
            raise HTTPException(status_code=404, detail="No predictions logged yet")

        current_data_raw = pd.read_csv(PREDICTION_DB).tail(n)
        
        if len(current_data_raw) == 0:
            raise HTTPException(status_code=404, detail="No predictions in database")

        current_data = pd.DataFrame({
            'mean_brightness': current_data_raw['mean_brightness'],
            'std_brightness': current_data_raw['std_brightness'],
            'max_brightness': current_data_raw['max_brightness'],
            'min_brightness': current_data_raw['min_brightness'],
            'contrast': current_data_raw['contrast'],
        })
        
        class_name_to_idx = {name: idx for idx, name in enumerate(class_names)}
        current_data['label'] = current_data_raw['predicted_class'].map(class_name_to_idx)

        train_data_path = Path("raw/train")
        if not train_data_path.exists():
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Monitoring Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .warning {{ background-color: #fff3cd; padding: 15px; border-radius: 5px; margin: 20px 0; }}
                    table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #4CAF50; color: white; }}
                </style>
            </head>
            <body>
                <h1>Monitoring Report</h1>
                <div class="warning">
                    <strong>⚠️ Warning:</strong> Training data not available at <code>{train_data_path}</code>.
                    Showing statistics for recent predictions only.
                </div>
                <h2>Recent Predictions Summary ({len(current_data)} samples)</h2>
                <table>
                    <tr>
                        <th>Feature</th>
                        <th>Mean</th>
                        <th>Std</th>
                        <th>Min</th>
                        <th>Max</th>
                    </tr>
                    <tr>
                        <td>Mean Brightness</td>
                        <td>{current_data['mean_brightness'].mean():.2f}</td>
                        <td>{current_data['mean_brightness'].std():.2f}</td>
                        <td>{current_data['mean_brightness'].min():.2f}</td>
                        <td>{current_data['mean_brightness'].max():.2f}</td>
                    </tr>
                    <tr>
                        <td>Std Brightness</td>
                        <td>{current_data['std_brightness'].mean():.2f}</td>
                        <td>{current_data['std_brightness'].std():.2f}</td>
                        <td>{current_data['std_brightness'].min():.2f}</td>
                        <td>{current_data['std_brightness'].max():.2f}</td>
                    </tr>
                    <tr>
                        <td>Contrast</td>
                        <td>{current_data['contrast'].mean():.2f}</td>
                        <td>{current_data['contrast'].std():.2f}</td>
                        <td>{current_data['contrast'].min():.2f}</td>
                        <td>{current_data['contrast'].max():.2f}</td>
                    </tr>
                </table>
                <h2>Class Distribution</h2>
                <table>
                    <tr>
                        <th>Class</th>
                        <th>Count</th>
                        <th>Percentage</th>
                    </tr>
            """
            
            for class_name, idx in class_name_to_idx.items():
                count = (current_data['label'] == idx).sum()
                percentage = (count / len(current_data)) * 100
                html_content += f"""
                    <tr>
                        <td>{class_name}</td>
                        <td>{count}</td>
                        <td>{percentage:.1f}%</td>
                    </tr>
                """
            
            html_content += """
                </table>
            </body>
            </html>
            """
            
            return HTMLResponse(content=html_content, status_code=200)

        from car_image_classification_using_cnn.drift_detection import extract_features_from_dataset, generate_drift_report

        print(f"Extracting features from {train_data_path}...")
        reference_data = extract_features_from_dataset(train_data_path, max_samples=n)
        print(f"Extracted {len(reference_data)} reference samples")

        report_path = Path("monitoring_report.html")
        print(f"Generating drift report...")
        generate_drift_report(reference_data, current_data, report_path)
        print(f"Report generated at {report_path}")

        with open(report_path) as f:
            html_content = f.read()

        return HTMLResponse(content=html_content, status_code=200)

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_detail = f"Failed to generate monitoring report: {str(e)}\n{traceback.format_exc()}"
        print(error_detail)
        raise HTTPException(status_code=500, detail=error_detail)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
