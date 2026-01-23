from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest
from PIL import Image

from car_image_classification_using_cnn.data_transform import convert_all_to_grayscale


class TestConvertAllToGrayscale:
    """Test suite for convert_all_to_grayscale function."""

    def test_creates_output_directory(self, tmp_path):
        """Test that output directory is created."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"

        input_dir.mkdir()

        convert_all_to_grayscale(input_dir, output_dir)

        assert output_dir.exists()
        assert output_dir.is_dir()

    def test_processes_jpg_images(self, tmp_path):
        """Test that JPG images are processed correctly."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"

        # Create input structure
        class_dir = input_dir / "TestClass"
        class_dir.mkdir(parents=True)

        # Create test image
        test_img = Image.new("RGB", (100, 100), color="red")
        test_img_path = class_dir / "test.jpg"
        test_img.save(test_img_path)

        convert_all_to_grayscale(input_dir, output_dir)

        # Check output exists
        output_class_dir = output_dir / "TestClass"
        assert output_class_dir.exists()

        output_img_path = output_class_dir / "test.jpg"
        assert output_img_path.exists()

        # Verify image is grayscale
        output_img = Image.open(output_img_path)
        assert output_img.mode == "L", "Output image should be grayscale (mode 'L')"

    def test_handles_multiple_classes(self, tmp_path):
        """Test that multiple class directories are processed."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"

        classes = ["Class1", "Class2", "Class3"]

        for class_name in classes:
            class_dir = input_dir / class_name
            class_dir.mkdir(parents=True)

            # Create test image
            test_img = Image.new("RGB", (50, 50), color="blue")
            test_img.save(class_dir / f"{class_name}.jpg")

        convert_all_to_grayscale(input_dir, output_dir)

        # Verify all classes were processed
        for class_name in classes:
            output_class_dir = output_dir / class_name
            assert output_class_dir.exists()
            assert (output_class_dir / f"{class_name}.jpg").exists()

    def test_handles_multiple_images_per_class(self, tmp_path):
        """Test that multiple images in a class are all processed."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"

        class_dir = input_dir / "TestClass"
        class_dir.mkdir(parents=True)

        # Create multiple test images
        image_names = ["img1.jpg", "img2.jpg", "img3.jpg"]
        for img_name in image_names:
            test_img = Image.new("RGB", (80, 80), color="green")
            test_img.save(class_dir / img_name)

        convert_all_to_grayscale(input_dir, output_dir)

        output_class_dir = output_dir / "TestClass"

        # Verify all images were processed
        for img_name in image_names:
            assert (output_class_dir / img_name).exists()

    def test_ignores_non_jpg_files(self, tmp_path):
        """Test that non-JPG files are ignored."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"

        class_dir = input_dir / "TestClass"
        class_dir.mkdir(parents=True)

        # Create a JPG and a non-JPG file
        jpg_img = Image.new("RGB", (50, 50), color="red")
        jpg_img.save(class_dir / "valid.jpg")

        # Create a text file
        (class_dir / "readme.txt").write_text("This is not an image")

        convert_all_to_grayscale(input_dir, output_dir)

        output_class_dir = output_dir / "TestClass"

        # Only JPG should be processed
        assert (output_class_dir / "valid.jpg").exists()
        assert not (output_class_dir / "readme.txt").exists()

    def test_handles_empty_class_directory(self, tmp_path):
        """Test handling of class directory with no images."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"

        # Create empty class directory
        class_dir = input_dir / "EmptyClass"
        class_dir.mkdir(parents=True)

        with patch("builtins.print") as mock_print:
            convert_all_to_grayscale(input_dir, output_dir)

            # Should print message about no images found
            print_calls = [str(call) for call in mock_print.call_args_list]
            assert any("No JPG images found" in str(call) for call in print_calls)

    def test_skips_files_in_root(self, tmp_path):
        """Test that files in root directory are ignored."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"

        input_dir.mkdir()

        # Create image in root (should be ignored)
        root_img = Image.new("RGB", (50, 50))
        root_img.save(input_dir / "root_image.jpg")

        # Create proper class directory
        class_dir = input_dir / "ValidClass"
        class_dir.mkdir()
        class_img = Image.new("RGB", (50, 50))
        class_img.save(class_dir / "class_image.jpg")

        convert_all_to_grayscale(input_dir, output_dir)

        # Root image should not be processed
        assert not (output_dir / "root_image.jpg").exists()

        # Class image should be processed
        assert (output_dir / "ValidClass" / "class_image.jpg").exists()

    def test_applies_histogram_equalization(self, tmp_path):
        """Test that histogram equalization is applied."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"

        class_dir = input_dir / "TestClass"
        class_dir.mkdir(parents=True)

        # Create a dark test image
        test_img = Image.new("RGB", (100, 100), color=(50, 50, 50))
        test_img.save(class_dir / "dark.jpg")

        with patch("car_image_classification_using_cnn.data_transform.ImageOps.equalize") as mock_equalize:
            # Make mock return a valid image
            mock_equalize.return_value = Image.new("L", (100, 100))

            convert_all_to_grayscale(input_dir, output_dir)

            # Verify equalize was called
            assert mock_equalize.called

    def test_preserves_image_names(self, tmp_path):
        """Test that image filenames are preserved."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"

        class_dir = input_dir / "TestClass"
        class_dir.mkdir(parents=True)

        original_names = ["image_1.jpg", "photo_abc.jpg", "test_123.jpg"]

        for name in original_names:
            img = Image.new("RGB", (50, 50))
            img.save(class_dir / name)

        convert_all_to_grayscale(input_dir, output_dir)

        output_class_dir = output_dir / "TestClass"

        for name in original_names:
            assert (output_class_dir / name).exists()

    def test_handles_path_objects(self, tmp_path):
        """Test that function accepts Path objects."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"

        input_dir.mkdir()
        class_dir = input_dir / "TestClass"
        class_dir.mkdir()

        test_img = Image.new("RGB", (50, 50))
        test_img.save(class_dir / "test.jpg")

        # Pass as Path objects (should work)
        convert_all_to_grayscale(Path(input_dir), Path(output_dir))

        assert output_dir.exists()

    def test_handles_string_paths(self, tmp_path):
        """Test that function accepts string paths."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"

        input_dir.mkdir()
        class_dir = input_dir / "TestClass"
        class_dir.mkdir()

        test_img = Image.new("RGB", (50, 50))
        test_img.save(class_dir / "test.jpg")

        # Pass as strings (should work)
        convert_all_to_grayscale(str(input_dir), str(output_dir))

        assert output_dir.exists()

    def test_prints_completion_message(self, tmp_path):
        """Test that completion message is printed."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"

        input_dir.mkdir()

        with patch("builtins.print") as mock_print:
            convert_all_to_grayscale(input_dir, output_dir)

            print_calls = [str(call) for call in mock_print.call_args_list]
            assert any("All images processed!" in str(call) for call in print_calls)
