"""Data drift detection for car classification model."""

from pathlib import Path
from typing import Tuple

import pandas as pd
import torch
from evidently.legacy.metric_preset import DataDriftPreset, DataQualityPreset, TargetDriftPreset
from evidently.legacy.report import Report
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

from car_image_classification_using_cnn.data import CarImageDataset
from car_image_classification_using_cnn.model import CarClassificationCNN


def extract_image_features(image_path: Path) -> dict[str, float]:
    """
    Extract statistical features from an image.
    
    Args:
        image_path: Path to the image file.
        
    Returns:
        Dictionary of extracted features.
    """
    img = Image.open(image_path).convert("RGB")
    img_array = torch.tensor(img).float()
    
    return {
        "mean_brightness": img_array.mean().item(),
        "std_brightness": img_array.std().item(),
        "max_brightness": img_array.max().item(),
        "min_brightness": img_array.min().item(),
        "contrast": (img_array.max() - img_array.min()).item(),
    }


def extract_features_from_dataset(data_path: Path, max_samples: int = 100) -> pd.DataFrame:
    """
    Extract features from a dataset of images.
    
    Args:
        data_path: Path to the dataset directory.
        max_samples: Maximum number of samples to process.
        
    Returns:
        DataFrame with extracted features and labels.
    """
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
    """
    Generate a data drift report comparing reference and current data.
    
    Args:
        reference_data: Reference dataset (training data).
        current_data: Current dataset (production data).
        output_path: Path to save the HTML report.
    """
    report = Report(metrics=[DataDriftPreset(), DataQualityPreset(), TargetDriftPreset()])
    
    report.run(reference_data=reference_data, current_data=current_data)
    report.save_html(str(output_path))
    
    print(f"Drift report saved to {output_path}")


if __name__ == "__main__":
    print("Extracting features from training data...")
    reference_data = extract_features_from_dataset(Path("raw/train"), max_samples=200)
    
    print("Extracting features from test data...")
    current_data = extract_features_from_dataset(Path("raw/test"), max_samples=100)
    
    print("Generating drift report...")
    generate_drift_report(reference_data, current_data, Path("reports/drift_report.html"))
    
    print("Done!")