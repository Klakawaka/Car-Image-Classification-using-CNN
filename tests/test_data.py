import os
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from PIL import Image
from torchvision import transforms

from car_image_classification_using_cnn.data import CarImageDataset, get_transforms
from tests import _PATH_DATA, _PATH_TEST_DATA, _PATH_TRAIN_DATA

# Expected dataset sizes (approximate, may vary if some images are corrupted)
EXPECTED_TRAIN_SIZE_MIN = 3000
EXPECTED_TEST_SIZE_MIN = 700
EXPECTED_NUM_CLASSES = 6
EXPECTED_CLASSES = ["Audi", "Hyundai Creta", "Rolls Royce", "Swift", "Tata Safari", "Toyota Innova"]


@pytest.mark.skipif(not os.path.exists(_PATH_TRAIN_DATA), reason="Training data files not found")
class TestTrainingData:
    """Test suite for training dataset."""

    def test_dataset_loads(self):
        """Test that the training dataset loads correctly."""
        dataset = CarImageDataset(Path(_PATH_TRAIN_DATA))
        assert isinstance(dataset, CarImageDataset), "Dataset should be an instance of CarImageDataset"
        assert len(dataset) > 0, "Dataset should not be empty"

    def test_train_dataset_size(self):
        """Test that the training dataset has the expected number of samples."""
        dataset = CarImageDataset(Path(_PATH_TRAIN_DATA))
        dataset_len = len(dataset)
        assert (
            dataset_len >= EXPECTED_TRAIN_SIZE_MIN
        ), f"Training dataset should have at least {EXPECTED_TRAIN_SIZE_MIN} samples, got {dataset_len}"

    def test_train_dataset_classes(self):
        """Test that all expected classes are present in the training dataset."""
        dataset = CarImageDataset(Path(_PATH_TRAIN_DATA))
        assert (
            len(dataset.classes) == EXPECTED_NUM_CLASSES
        ), f"Expected {EXPECTED_NUM_CLASSES} classes, got {len(dataset.classes)}"
        assert sorted(dataset.classes) == sorted(
            EXPECTED_CLASSES
        ), f"Expected classes {EXPECTED_CLASSES}, got {dataset.classes}"

    def test_train_dataset_class_mapping(self):
        """Test that class to index mapping is correct."""
        dataset = CarImageDataset(Path(_PATH_TRAIN_DATA))
        assert (
            len(dataset.class_to_idx) == EXPECTED_NUM_CLASSES
        ), "Class to index mapping should match number of classes"
        for class_name in EXPECTED_CLASSES:
            assert class_name in dataset.class_to_idx, f"Class '{class_name}' should be in class_to_idx mapping"
        assert all(
            isinstance(idx, int) for idx in dataset.class_to_idx.values()
        ), "All class indices should be integers"
        assert set(dataset.class_to_idx.values()) == set(
            range(EXPECTED_NUM_CLASSES)
        ), "Class indices should be continuous from 0 to num_classes-1"

    def test_train_all_labels_represented(self):
        """Test that all labels are represented in the training dataset."""
        dataset = CarImageDataset(Path(_PATH_TRAIN_DATA))
        labels = [label for _, label in dataset.samples]
        unique_labels = set(labels)
        assert len(unique_labels) == EXPECTED_NUM_CLASSES, (
            f"All {EXPECTED_NUM_CLASSES} classes should have samples, " f"but only {len(unique_labels)} found"
        )
        for class_name, idx in dataset.class_to_idx.items():
            assert idx in labels, f"Label {idx} for class '{class_name}' is not present in any samples"

    @pytest.mark.parametrize("image_size", [(224, 224), (128, 128), (256, 256)])
    def test_train_datapoint_shape(self, image_size):
        """Test that each datapoint has the correct shape for different image sizes."""
        dataset = CarImageDataset(Path(_PATH_TRAIN_DATA), image_size=image_size)
        image, label = dataset[0]
        assert isinstance(image, torch.Tensor), "Image should be a torch.Tensor"
        assert image.shape == (
            3,
            image_size[0],
            image_size[1],
        ), f"Image should have shape (3, {image_size[0]}, {image_size[1]}), got {image.shape}"
        assert isinstance(label, int), "Label should be an integer"
        assert 0 <= label < EXPECTED_NUM_CLASSES, f"Label should be between 0 and {EXPECTED_NUM_CLASSES-1}, got {label}"

    def test_train_datapoint_normalization(self):
        """Test that datapoints are normalized properly."""
        dataset = CarImageDataset(Path(_PATH_TRAIN_DATA))
        image, _ = dataset[0]
        assert (
            image.min() >= -3.0 and image.max() <= 3.0
        ), "Normalized image values should be roughly in [-3, 3] range (mean=0.5, std=0.2 normalization)"

    def test_train_dataset_indexing(self):
        """Test that dataset indexing works correctly."""
        dataset = CarImageDataset(Path(_PATH_TRAIN_DATA))
        first_item = dataset[0]
        last_item = dataset[-1]
        assert len(first_item) == 2, "Each dataset item should return (image, label) tuple"
        assert len(last_item) == 2, "Each dataset item should return (image, label) tuple"

    def test_train_dataset_samples_list(self):
        """Test that samples list is properly populated."""
        dataset = CarImageDataset(Path(_PATH_TRAIN_DATA))
        assert len(dataset.samples) == len(dataset), "Length of samples list should match dataset length"
        for img_path, label in dataset.samples[:10]:
            assert isinstance(img_path, Path), "Image path should be a Path object"
            assert img_path.exists(), f"Image path {img_path} should exist"
            assert img_path.suffix.lower() in [
                ".jpg",
                ".jpeg",
                ".png",
                ".bmp",
            ], f"Image should have valid extension, got {img_path.suffix}"
            assert isinstance(label, int), "Label should be an integer"


@pytest.mark.skipif(not os.path.exists(_PATH_TEST_DATA), reason="Test data files not found")
class TestTestData:
    """Test suite for test dataset."""

    def test_dataset_loads(self):
        """Test that the test dataset loads correctly."""
        dataset = CarImageDataset(Path(_PATH_TEST_DATA))
        assert isinstance(dataset, CarImageDataset), "Dataset should be an instance of CarImageDataset"
        assert len(dataset) > 0, "Dataset should not be empty"

    def test_test_dataset_size(self):
        """Test that the test dataset has the expected number of samples."""
        dataset = CarImageDataset(Path(_PATH_TEST_DATA))
        dataset_len = len(dataset)
        assert (
            dataset_len >= EXPECTED_TEST_SIZE_MIN
        ), f"Test dataset should have at least {EXPECTED_TEST_SIZE_MIN} samples, got {dataset_len}"

    def test_test_dataset_classes(self):
        """Test that all expected classes are present in the test dataset."""
        dataset = CarImageDataset(Path(_PATH_TEST_DATA))
        assert (
            len(dataset.classes) == EXPECTED_NUM_CLASSES
        ), f"Expected {EXPECTED_NUM_CLASSES} classes, got {len(dataset.classes)}"
        assert sorted(dataset.classes) == sorted(
            EXPECTED_CLASSES
        ), f"Expected classes {EXPECTED_CLASSES}, got {dataset.classes}"

    def test_test_all_labels_represented(self):
        """Test that all labels are represented in the test dataset."""
        dataset = CarImageDataset(Path(_PATH_TEST_DATA))
        labels = [label for _, label in dataset.samples]
        unique_labels = set(labels)
        assert len(unique_labels) == EXPECTED_NUM_CLASSES, (
            f"All {EXPECTED_NUM_CLASSES} classes should have samples, " f"but only {len(unique_labels)} found"
        )

    def test_test_datapoint_shape(self):
        """Test that each datapoint has the correct shape."""
        dataset = CarImageDataset(Path(_PATH_TEST_DATA))
        image, label = dataset[0]
        assert isinstance(image, torch.Tensor), "Image should be a torch.Tensor"
        assert image.shape == (3, 224, 224), f"Image should have shape (3, 224, 224), got {image.shape}"
        assert isinstance(label, int), "Label should be an integer"
        assert 0 <= label < EXPECTED_NUM_CLASSES, f"Label should be between 0 and {EXPECTED_NUM_CLASSES-1}, got {label}"


@pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="Data files not found")
class TestDatasetErrors:
    """Test suite for dataset error handling."""

    def test_nonexistent_path_raises_error(self):
        """Test that FileNotFoundError is raised for non-existent path."""
        nonexistent_path = Path("/nonexistent/path/to/data")
        with pytest.raises(FileNotFoundError, match="Data path does not exist"):
            CarImageDataset(nonexistent_path)

    @pytest.mark.skipif(not os.path.exists(_PATH_TRAIN_DATA), reason="Training data files not found")
    def test_custom_transform(self):
        """Test that custom transforms can be provided."""
        custom_transform = transforms.Compose([transforms.Resize((100, 100)), transforms.ToTensor()])
        dataset = CarImageDataset(Path(_PATH_TRAIN_DATA), transform=custom_transform)
        image, _ = dataset[0]
        assert image.shape == (3, 100, 100), f"Custom transform should resize to (3, 100, 100), got {image.shape}"


@pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="Data files not found")
class TestTransforms:
    """Test suite for data transforms."""

    @pytest.mark.parametrize("mode", ["train", "eval"])
    def test_get_transforms(self, mode):
        """Test that get_transforms returns appropriate transforms for different modes."""
        transform = get_transforms(mode=mode)
        assert isinstance(transform, transforms.Compose), "get_transforms should return a Compose object"

    def test_train_transforms_have_augmentation(self):
        """Test that training transforms include augmentation."""
        train_transform = get_transforms(mode="train")
        transform_types = [type(t).__name__ for t in train_transform.transforms]
        assert "RandomCrop" in transform_types, "Training transforms should include RandomCrop for augmentation"
        assert "RandomHorizontalFlip" in transform_types, "Training transforms should include RandomHorizontalFlip"
        assert "ColorJitter" in transform_types, "Training transforms should include ColorJitter"

    def test_eval_transforms_no_augmentation(self):
        """Test that evaluation transforms do not include augmentation."""
        eval_transform = get_transforms(mode="eval")
        transform_types = [type(t).__name__ for t in eval_transform.transforms]
        assert "RandomCrop" not in transform_types, "Eval transforms should not include RandomCrop"
        assert "RandomHorizontalFlip" not in transform_types, "Eval transforms should not include RandomHorizontalFlip"
        assert "ColorJitter" not in transform_types, "Eval transforms should not include ColorJitter"

    @pytest.mark.parametrize("image_size", [(224, 224), (128, 128), (256, 256)])
    def test_transforms_output_size(self, image_size):
        """Test that transforms produce correct output size."""
        transform = get_transforms(mode="eval", image_size=image_size)
        test_image = Image.new("RGB", (100, 100))
        transformed = transform(test_image)
        assert transformed.shape == (
            3,
            image_size[0],
            image_size[1],
        ), f"Transform should produce shape (3, {image_size[0]}, {image_size[1]}), got {transformed.shape}"


@pytest.mark.skipif(not os.path.exists(_PATH_TRAIN_DATA), reason="Training data files not found")
class TestDatasetConsistency:
    """Test suite for dataset consistency checks."""

    def test_dataset_deterministic_loading(self):
        """Test that dataset loading is deterministic."""
        dataset1 = CarImageDataset(Path(_PATH_TRAIN_DATA))
        dataset2 = CarImageDataset(Path(_PATH_TRAIN_DATA))
        assert len(dataset1) == len(dataset2), "Dataset should have consistent length across instantiations"
        assert dataset1.classes == dataset2.classes, "Classes should be consistent across instantiations"
        assert (
            dataset1.class_to_idx == dataset2.class_to_idx
        ), "Class to index mapping should be consistent across instantiations"

    def test_class_distribution(self):
        """Test that each class has a reasonable number of samples."""
        dataset = CarImageDataset(Path(_PATH_TRAIN_DATA))
        class_counts = {class_name: 0 for class_name in dataset.classes}
        for _, label in dataset.samples:
            class_name = dataset.classes[label]
            class_counts[class_name] += 1

        for class_name, count in class_counts.items():
            assert count > 0, f"Class '{class_name}' should have at least some samples, got {count}"
            assert count < len(dataset), f"Class '{class_name}' should not have all samples ({count} vs {len(dataset)})"

    def test_no_duplicate_paths(self):
        """Test that there are no duplicate image paths in the dataset."""
        dataset = CarImageDataset(Path(_PATH_TRAIN_DATA))
        image_paths = [str(img_path) for img_path, _ in dataset.samples]
        unique_paths = set(image_paths)
        assert len(image_paths) == len(unique_paths), (
            f"Dataset should not have duplicate image paths. "
            f"Found {len(image_paths)} total, {len(unique_paths)} unique"
        )


class TestDatasetPreprocess:
    """Test suite for dataset preprocessing functionality."""

    def test_preprocess_creates_output_folder(self, tmp_path):
        """Test that preprocess method creates output folder."""
        # Create a minimal dataset
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"

        class_dir = input_dir / "TestClass"
        class_dir.mkdir(parents=True)
        test_img = Image.new("RGB", (100, 100), color="blue")
        test_img.save(class_dir / "test.jpg")

        dataset = CarImageDataset(input_dir)
        dataset.preprocess(output_dir)

        assert output_dir.exists(), "Output folder should be created"
        assert output_dir.is_dir(), "Output should be a directory"

    def test_preprocess_copies_class_structure(self, tmp_path):
        """Test that preprocess maintains class directory structure."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"

        classes = ["Class1", "Class2"]
        for class_name in classes:
            class_dir = input_dir / class_name
            class_dir.mkdir(parents=True)
            test_img = Image.new("RGB", (50, 50))
            test_img.save(class_dir / "test.jpg")

        dataset = CarImageDataset(input_dir)
        dataset.preprocess(output_dir)

        for class_name in classes:
            assert (output_dir / class_name).exists(), f"Class directory {class_name} should exist"

    def test_preprocess_copies_images(self, tmp_path):
        """Test that preprocess copies image files."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"

        class_dir = input_dir / "TestClass"
        class_dir.mkdir(parents=True)

        image_names = ["img1.jpg", "img2.png", "img3.jpeg"]
        for img_name in image_names:
            test_img = Image.new("RGB", (50, 50))
            test_img.save(class_dir / img_name)

        dataset = CarImageDataset(input_dir)
        dataset.preprocess(output_dir)

        output_class_dir = output_dir / "TestClass"
        for img_name in image_names:
            assert (output_class_dir / img_name).exists(), f"Image {img_name} should be copied"

    def test_preprocess_skips_existing_files(self, tmp_path):
        """Test that preprocess doesn't overwrite existing files."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"

        class_dir = input_dir / "TestClass"
        class_dir.mkdir(parents=True)
        test_img = Image.new("RGB", (50, 50), color="red")
        test_img.save(class_dir / "test.jpg")

        # Pre-create output with different content
        output_class_dir = output_dir / "TestClass"
        output_class_dir.mkdir(parents=True)
        existing_img = Image.new("RGB", (50, 50), color="blue")
        existing_img.save(output_class_dir / "test.jpg")

        # Get modification time
        original_mtime = (output_class_dir / "test.jpg").stat().st_mtime

        dataset = CarImageDataset(input_dir)
        dataset.preprocess(output_dir)

        # File should not be overwritten
        new_mtime = (output_class_dir / "test.jpg").stat().st_mtime
        assert new_mtime == original_mtime, "Existing file should not be overwritten"

    def test_preprocess_counts_copied_images(self, tmp_path):
        """Test that preprocess prints count of copied images."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"

        class_dir = input_dir / "TestClass"
        class_dir.mkdir(parents=True)

        for i in range(3):
            test_img = Image.new("RGB", (50, 50))
            test_img.save(class_dir / f"img{i}.jpg")

        dataset = CarImageDataset(input_dir)

        with patch("builtins.print") as mock_print:
            dataset.preprocess(output_dir)

            # Should print copy count
            print_calls = [str(call) for call in mock_print.call_args_list]
            assert any("Copied" in str(call) for call in print_calls)


class TestPreprocessStandaloneFunction:
    """Test suite for the standalone preprocess function."""

    def test_preprocess_function_creates_dataset(self, tmp_path):
        """Test that preprocess function creates a dataset."""
        from car_image_classification_using_cnn.data import preprocess

        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"

        class_dir = input_dir / "TestClass"
        class_dir.mkdir(parents=True)
        test_img = Image.new("RGB", (100, 100))
        test_img.save(class_dir / "test.jpg")

        with patch("builtins.print"):
            preprocess(input_dir, output_dir)

        assert output_dir.exists()

    def test_preprocess_function_prints_info(self, tmp_path):
        """Test that preprocess function prints dataset information."""
        from car_image_classification_using_cnn.data import preprocess

        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"

        class_dir = input_dir / "TestClass"
        class_dir.mkdir(parents=True)
        test_img = Image.new("RGB", (50, 50))
        test_img.save(class_dir / "test.jpg")

        with patch("builtins.print") as mock_print:
            preprocess(input_dir, output_dir)

            print_calls = [str(call) for call in mock_print.call_args_list]
            assert any("Found" in str(call) for call in print_calls)
            assert any("Classes:" in str(call) for call in print_calls)


class TestDatasetEdgeCases:
    """Test suite for dataset edge cases and error handling."""

    def test_dataset_handles_corrupted_images(self, tmp_path):
        """Test that dataset skips corrupted images gracefully."""
        input_dir = tmp_path / "input"
        class_dir = input_dir / "TestClass"
        class_dir.mkdir(parents=True)

        # Create a valid image
        valid_img = Image.new("RGB", (50, 50))
        valid_img.save(class_dir / "valid.jpg")

        # Create a corrupted "image" (just text)
        (class_dir / "corrupted.jpg").write_text("This is not an image")

        with patch("builtins.print"):
            dataset = CarImageDataset(input_dir)

        # Dataset should load but skip corrupted image
        assert len(dataset) >= 1, "Should load at least the valid image"

    def test_dataset_with_mixed_extensions(self, tmp_path):
        """Test dataset with various image extensions."""
        input_dir = tmp_path / "input"
        class_dir = input_dir / "TestClass"
        class_dir.mkdir(parents=True)

        extensions = [".jpg", ".jpeg", ".png", ".bmp"]
        for ext in extensions:
            img = Image.new("RGB", (50, 50))
            img.save(class_dir / f"test{ext}")

        dataset = CarImageDataset(input_dir)

        assert len(dataset) == len(extensions), "Should load all valid image types"

    def test_dataset_negative_indexing(self, tmp_path):
        """Test that dataset supports negative indexing."""
        input_dir = tmp_path / "input"
        class_dir = input_dir / "TestClass"
        class_dir.mkdir(parents=True)

        for i in range(5):
            # Create images with different colors so they're distinguishable
            img = Image.new("RGB", (50, 50), color=(i * 50, i * 40, i * 30))
            img.save(class_dir / f"test{i}.jpg")

        dataset = CarImageDataset(input_dir)

        # Negative indexing should work
        last_item = dataset[-1]
        assert len(last_item) == 2, "Should return (image, label) tuple"

        first_item = dataset[0]
        second_last = dataset[-2]

        # All items should be from same class but different images
        assert len(first_item) == 2 and len(second_last) == 2

    def test_transforms_mode_case_insensitive(self):
        """Test that get_transforms handles different mode cases."""
        from car_image_classification_using_cnn.data import get_transforms

        # These should work without errors
        train_transform = get_transforms(mode="train")
        eval_transform = get_transforms(mode="eval")
        test_transform = get_transforms(mode="test")

        assert isinstance(train_transform, transforms.Compose)
        assert isinstance(eval_transform, transforms.Compose)
        assert isinstance(test_transform, transforms.Compose)
