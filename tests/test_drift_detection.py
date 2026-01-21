"""Tests for drift detection functionality."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from car_image_classification_using_cnn.drift_detection import (
    extract_features_from_dataset,
    extract_image_features,
    generate_drift_report,
)
from tests import _PATH_TEST_DATA, _PATH_TRAIN_DATA


class TestExtractImageFeatures:

    def test_extract_features_from_uniform_image(self, tmp_path: Path) -> None:
        test_img = Image.new("RGB", (224, 224), color=(128, 128, 128))
        img_path = tmp_path / "gray.jpg"
        test_img.save(img_path)

        features = extract_image_features(img_path)

        assert isinstance(features, dict), "Features should be a dictionary"
        assert "mean_brightness" in features
        assert "std_brightness" in features
        assert "max_brightness" in features
        assert "min_brightness" in features
        assert "contrast" in features

    def test_extract_features_values_uniform_image(self, tmp_path: Path) -> None:
        """Test that extracted features have reasonable values for uniform image."""
        test_img = Image.new("RGB", (224, 224), color=(128, 128, 128))
        img_path = tmp_path / "gray.jpg"
        test_img.save(img_path)

        features = extract_image_features(img_path)

        assert 100 < features["mean_brightness"] < 150
        assert features["std_brightness"] < 10
        assert features["contrast"] < 20

    def test_extract_features_black_image(self, tmp_path: Path) -> None:
        """Test feature extraction from black image."""
        black_img = Image.new("RGB", (224, 224), color=(0, 0, 0))
        img_path = tmp_path / "black.jpg"
        black_img.save(img_path)

        features = extract_image_features(img_path)

        assert features["mean_brightness"] < 50
        assert features["min_brightness"] < 10
        assert features["contrast"] < 50

    def test_extract_features_white_image(self, tmp_path: Path) -> None:
        """Test feature extraction from white image."""
        white_img = Image.new("RGB", (224, 224), color=(255, 255, 255))
        img_path = tmp_path / "white.jpg"
        white_img.save(img_path)

        features = extract_image_features(img_path)

        assert features["mean_brightness"] > 200
        assert features["max_brightness"] > 240
        assert features["contrast"] < 50

    def test_extract_features_brightness_comparison(self, tmp_path: Path) -> None:
        """Test that white image has higher brightness than black image."""
        black_img = Image.new("RGB", (224, 224), color=(0, 0, 0))
        black_path = tmp_path / "black.jpg"
        black_img.save(black_path)

        white_img = Image.new("RGB", (224, 224), color=(255, 255, 255))
        white_path = tmp_path / "white.jpg"
        white_img.save(white_path)

        black_features = extract_image_features(black_path)
        white_features = extract_image_features(white_path)

        assert white_features["mean_brightness"] > black_features["mean_brightness"]
        assert white_features["max_brightness"] > black_features["max_brightness"]

    def test_extract_features_all_values_are_floats(self, tmp_path: Path) -> None:
        """Test that all extracted features are float values."""
        test_img = Image.new("RGB", (224, 224), color=(100, 150, 200))
        img_path = tmp_path / "test.jpg"
        test_img.save(img_path)

        features = extract_image_features(img_path)

        for key, value in features.items():
            assert isinstance(value, float), f"{key} should be float, got {type(value)}"

    def test_extract_features_different_image_sizes(self, tmp_path: Path) -> None:
        """Test feature extraction works with different image sizes."""
        for size in [(100, 100), (224, 224), (512, 512)]:
            test_img = Image.new("RGB", size, color=(128, 128, 128))
            img_path = tmp_path / f"test_{size[0]}x{size[1]}.jpg"
            test_img.save(img_path)

            features = extract_image_features(img_path)
            assert len(features) == 5, f"Should extract 5 features for {size} image"


@pytest.mark.skipif(
    not os.path.exists(_PATH_TRAIN_DATA) or not os.path.exists(_PATH_TEST_DATA),
    reason="Training or test data not found",
)
class TestExtractFeaturesFromDataset:
    """Test suite for extract_features_from_dataset function."""

    def test_extract_features_returns_dataframe(self) -> None:
        """Test that function returns a pandas DataFrame."""
        df = extract_features_from_dataset(Path(_PATH_TRAIN_DATA), max_samples=10)
        assert isinstance(df, pd.DataFrame)

    def test_extract_features_respects_max_samples(self) -> None:
        """Test that max_samples parameter is respected."""
        df = extract_features_from_dataset(Path(_PATH_TRAIN_DATA), max_samples=10)
        assert len(df) <= 10

    def test_extract_features_has_required_columns(self) -> None:
        """Test that DataFrame has all required columns."""
        df = extract_features_from_dataset(Path(_PATH_TRAIN_DATA), max_samples=5)

        required_columns = [
            "mean_brightness",
            "std_brightness",
            "max_brightness",
            "min_brightness",
            "contrast",
            "label",
        ]
        for col in required_columns:
            assert col in df.columns, f"DataFrame should have column '{col}'"

    def test_extract_features_no_nan_values(self) -> None:
        """Test that extracted features have no NaN values."""
        df = extract_features_from_dataset(Path(_PATH_TRAIN_DATA), max_samples=10)

        for col in df.columns:
            assert df[col].notna().all(), f"Column '{col}' should not have NaN values"

    def test_extract_features_from_small_dataset(self, tmp_path: Path) -> None:
        """Test extraction when max_samples exceeds dataset size."""
        data_dir = tmp_path / "small_dataset"
        data_dir.mkdir()
        class_dir = data_dir / "class1"
        class_dir.mkdir()

        for i in range(3):
            img = Image.new("RGB", (224, 224), color=(100, 100, 100))
            img.save(class_dir / f"img_{i}.jpg")

        df = extract_features_from_dataset(data_dir, max_samples=100)
        assert len(df) == 3

    def test_extract_features_consistency(self) -> None:
        """Test that feature extraction is consistent across runs."""
        df1 = extract_features_from_dataset(Path(_PATH_TRAIN_DATA), max_samples=5)
        df2 = extract_features_from_dataset(Path(_PATH_TRAIN_DATA), max_samples=5)

        assert len(df1) == len(df2)
        for col in ["mean_brightness", "std_brightness", "contrast"]:
            assert df1[col].equals(df2[col]), f"{col} should be consistent across runs"

    def test_extract_features_label_values(self) -> None:
        """Test that label values are valid integers."""
        df = extract_features_from_dataset(Path(_PATH_TRAIN_DATA), max_samples=20)

        assert df["label"].dtype in [np.int32, np.int64], "Labels should be integers"
        assert df["label"].min() >= 0, "Labels should be non-negative"

    def test_extract_features_brightness_range(self) -> None:
        """Test that brightness values are within valid range."""
        df = extract_features_from_dataset(Path(_PATH_TRAIN_DATA), max_samples=20)

        assert (df["mean_brightness"] >= 0).all(), "Mean brightness should be non-negative"
        assert (df["mean_brightness"] <= 255).all(), "Mean brightness should not exceed 255"
        assert (df["min_brightness"] >= 0).all(), "Min brightness should be non-negative"
        assert (df["max_brightness"] <= 255).all(), "Max brightness should not exceed 255"


class TestGenerateDriftReport:

    def test_generate_report_creates_file(self, tmp_path: Path) -> None:
        """Test that drift report file is created."""
        reference_data = pd.DataFrame(
            {
                "mean_brightness": [100, 110, 120, 130],
                "std_brightness": [10, 15, 20, 25],
                "contrast": [50, 60, 70, 80],
                "label": [0, 1, 2, 3],
            }
        )

        current_data = pd.DataFrame(
            {
                "mean_brightness": [105, 115, 125, 135],
                "std_brightness": [12, 17, 22, 27],
                "contrast": [55, 65, 75, 85],
                "label": [0, 1, 2, 3],
            }
        )

        output_path = tmp_path / "drift_report.html"
        generate_drift_report(reference_data, current_data, output_path)

        assert output_path.exists(), "Drift report file should be created"

    def test_generate_report_not_empty(self, tmp_path: Path) -> None:
        """Test that generated report is not empty."""
        reference_data = pd.DataFrame(
            {
                "mean_brightness": [100, 110, 120],
                "std_brightness": [10, 15, 20],
                "contrast": [50, 60, 70],
                "label": [0, 1, 2],
            }
        )

        current_data = pd.DataFrame(
            {
                "mean_brightness": [105, 115, 125],
                "std_brightness": [12, 17, 22],
                "contrast": [55, 65, 75],
                "label": [0, 1, 2],
            }
        )

        output_path = tmp_path / "drift_report.html"
        generate_drift_report(reference_data, current_data, output_path)

        assert output_path.stat().st_size > 0, "Drift report should not be empty"

    @pytest.mark.skipif(
        not os.path.exists(_PATH_TRAIN_DATA) or not os.path.exists(_PATH_TEST_DATA),
        reason="Training or test data not found",
    )
    def test_generate_report_with_real_data(self, tmp_path: Path) -> None:
        train_features = extract_features_from_dataset(Path(_PATH_TRAIN_DATA), max_samples=30)
        test_features = extract_features_from_dataset(Path(_PATH_TEST_DATA), max_samples=30)

        output_path = tmp_path / "real_drift_report.html"
        generate_drift_report(train_features, test_features, output_path)

        assert output_path.exists()
        assert output_path.stat().st_size > 1000

    def test_generate_report_with_identical_data(self, tmp_path: Path) -> None:
        data = pd.DataFrame(
            {
                "mean_brightness": [100, 110, 120],
                "std_brightness": [10, 15, 20],
                "contrast": [50, 60, 70],
                "label": [0, 1, 2],
            }
        )

        output_path = tmp_path / "no_drift_report.html"
        generate_drift_report(data, data.copy(), output_path)

        assert output_path.exists()


@pytest.mark.skipif(
    not os.path.exists(_PATH_TRAIN_DATA) or not os.path.exists(_PATH_TEST_DATA),
    reason="Training or test data not found",
)
class TestDriftDetectionIntegration:

    def test_complete_drift_detection_workflow(self, tmp_path: Path) -> None:
        train_features = extract_features_from_dataset(Path(_PATH_TRAIN_DATA), max_samples=20)
        test_features = extract_features_from_dataset(Path(_PATH_TEST_DATA), max_samples=20)

        assert len(train_features) > 0
        assert len(test_features) > 0

        output_path = tmp_path / "drift_report.html"
        generate_drift_report(train_features, test_features, output_path)

        assert output_path.exists()

    def test_drift_detection_different_distributions(self) -> None:
        train_features = extract_features_from_dataset(Path(_PATH_TRAIN_DATA), max_samples=30)
        test_features = extract_features_from_dataset(Path(_PATH_TEST_DATA), max_samples=30)

        train_mean = train_features["mean_brightness"].mean()
        test_mean = test_features["mean_brightness"].mean()

        assert isinstance(train_mean, (float, np.floating))
        assert isinstance(test_mean, (float, np.floating))

    def test_feature_coverage_across_datasets(self) -> None:
        train_features = extract_features_from_dataset(Path(_PATH_TRAIN_DATA), max_samples=10)
        test_features = extract_features_from_dataset(Path(_PATH_TEST_DATA), max_samples=10)

        assert set(train_features.columns) == set(test_features.columns)

    def test_feature_statistics_validity(self) -> None:
        features = extract_features_from_dataset(Path(_PATH_TRAIN_DATA), max_samples=50)

        assert features["std_brightness"].min() >= 0, "Std should be non-negative"
        assert features["contrast"].min() >= 0, "Contrast should be non-negative"
        assert features["min_brightness"].min() >= 0, "Min brightness should be non-negative"
        assert features["max_brightness"].max() <= 255, "Max brightness should not exceed 255"


class TestDriftDetectionEdgeCases:

    def test_extract_features_empty_dataset(self, tmp_path: Path) -> None:
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        class_dir = empty_dir / "class1"
        class_dir.mkdir()

        df = extract_features_from_dataset(empty_dir, max_samples=10)
        assert len(df) == 0, "Should return empty DataFrame for empty dataset"

    def test_extract_features_single_image(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "single"
        data_dir.mkdir()
        class_dir = data_dir / "class1"
        class_dir.mkdir()

        img = Image.new("RGB", (224, 224), color=(128, 128, 128))
        img.save(class_dir / "img.jpg")

        df = extract_features_from_dataset(data_dir, max_samples=10)
        assert len(df) == 1

    def test_generate_report_minimal_data(self, tmp_path: Path) -> None:
        minimal_data = pd.DataFrame(
            {
                "mean_brightness": [100],
                "std_brightness": [10],
                "contrast": [50],
                "label": [0],
            }
        )

        output_path = tmp_path / "minimal_report.html"
        generate_drift_report(minimal_data, minimal_data, output_path)

        assert output_path.exists()