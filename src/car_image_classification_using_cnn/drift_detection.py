from pathlib import Path

import numpy as np
import pandas as pd
from evidently.legacy.report import Report
from evidently.legacy.metric_preset import DataDriftPreset, DataQualityPreset

from PIL import Image

from car_image_classification_using_cnn.data import CarImageDataset


def extract_image_features(image_path: Path) -> dict[str, float]:
    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img, dtype=np.float32)

    return {
        "mean_brightness": float(img_array.mean()),
        "std_brightness": float(img_array.std()),
        "max_brightness": float(img_array.max()),
        "min_brightness": float(img_array.min()),
        "contrast": float(img_array.max() - img_array.min()),
    }


def extract_features_from_dataset(data_path: Path, max_samples: int = 100) -> pd.DataFrame:
    dataset = CarImageDataset(data_path)
    features_list = []

    for idx in range(min(len(dataset), max_samples)):
        img_path = dataset.samples[idx][0]
        label = dataset.samples[idx][1]

        features = extract_image_features(img_path)
        features["label"] = label
        features_list.append(features)

    return pd.DataFrame(features_list)


def generate_drift_report(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    output_path: Path = Path("drift_report.html"),
) -> None:
    report = Report(
        metrics=[
            DataDriftPreset(),
            DataQualityPreset(),
        ]
    )

    report.run(reference_data=reference_data, current_data=current_data)
    report.save_html(str(output_path))

    print(f"Drift report saved to {output_path}")


if __name__ == "__main__":
    print("Extracting features from training data...")
    reference_data = extract_features_from_dataset(Path("raw/train"), max_samples=200)

    print("Extracting features from test data...")
    current_data = extract_features_from_dataset(Path("raw/test"), max_samples=100)

    print("Generating drift report...")
    output_dir = Path("reports")
    output_dir.mkdir(exist_ok=True)
    generate_drift_report(reference_data, current_data, output_dir / "drift_report.html")

    print("Done!")
