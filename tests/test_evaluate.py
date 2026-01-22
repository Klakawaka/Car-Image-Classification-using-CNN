import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from car_image_classification_using_cnn.evaluate import evaluate
from tests import _PATH_TEST_DATA


@pytest.fixture
def mock_model_checkpoint(tmp_path):
    """Create a mock model checkpoint for testing."""
    model_path = tmp_path / "test_model.pth"
    
    # Create a simple checkpoint
    checkpoint = {
        "model_state_dict": {
            "conv1.weight": torch.randn(64, 3, 3, 3),
            "conv1.bias": torch.randn(64),
        },
        "classes": ["Audi", "Hyundai Creta", "Rolls Royce", "Swift", "Tata Safari", "Toyota Innova"],
        "epoch": 10,
        "val_acc": 85.5,
    }
    
    torch.save(checkpoint, model_path)
    return model_path


@pytest.fixture
def simple_model_checkpoint(tmp_path):
    """Create a checkpoint with actual model state dict."""
    from car_image_classification_using_cnn.model import create_model
    
    model_path = tmp_path / "simple_model.pth"
    classes = ["Audi", "Hyundai Creta", "Rolls Royce", "Swift", "Tata Safari", "Toyota Innova"]
    
    model = create_model(num_classes=len(classes))
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "classes": classes,
        "epoch": 5,
        "val_acc": 75.0,
    }
    
    torch.save(checkpoint, model_path)
    return model_path, classes


class TestEvaluateFunction:
    """Test suite for the evaluate function."""
    
    def test_evaluate_with_real_data(self, simple_model_checkpoint, tmp_path):
        """Test that evaluate can run with real data."""
        model_path, classes = simple_model_checkpoint
        
        # Create a minimal test dataset
        test_dir = tmp_path / "test_data"
        for class_name in classes[:3]:  # Just use first 3 classes
            class_dir = test_dir / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            # Create a dummy image
            from PIL import Image
            img = Image.new("RGB", (100, 100), color="red")
            img.save(class_dir / "test.jpg")
        
        # Should run without errors (might print warning about class mismatch)
        evaluate(
            model_path=model_path,
            test_data_dir=test_dir,
            batch_size=2,
            device="cpu"
        )
    
    def test_evaluate_raises_error_for_missing_classes(self, tmp_path):
        """Test that evaluate raises error when checkpoint has no classes."""
        model_path = tmp_path / "bad_model.pth"
        checkpoint = {
            "model_state_dict": {},
            # Missing 'classes' key
        }
        torch.save(checkpoint, model_path)
        
        with pytest.raises(ValueError, match="Checkpoint does not contain class information"):
            evaluate(
                model_path=model_path,
                test_data_dir=Path("dummy"),
                batch_size=2,
                device="cpu"
            )
    
    def test_evaluate_warns_on_class_mismatch(self, simple_model_checkpoint, tmp_path):
        """Test that evaluate warns when test classes don't match training classes."""
        model_path, classes = simple_model_checkpoint
        
        # Create test directory with different classes
        test_dir = tmp_path / "test_data"
        different_classes = ["ClassA", "ClassB", "ClassC"]
        for class_name in different_classes:
            class_dir = test_dir / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            from PIL import Image
            img = Image.new("RGB", (100, 100), color="blue")
            img.save(class_dir / "test.jpg")
        
        with patch("builtins.print") as mock_print:
            evaluate(
                model_path=model_path,
                test_data_dir=test_dir,
                batch_size=2,
                device="cpu"
            )
            
            # Check that warning was printed
            print_calls = [str(call) for call in mock_print.call_args_list]
            assert any("WARNING" in str(call) for call in print_calls)
    
    def test_evaluate_with_same_classes(self, simple_model_checkpoint, tmp_path):
        """Test evaluate with matching training and test classes."""
        model_path, classes = simple_model_checkpoint
        
        test_dir = tmp_path / "test_data"
        # Create test data with all classes
        for class_name in classes:
            class_dir = test_dir / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            from PIL import Image
            img = Image.new("RGB", (100, 100))
            img.save(class_dir / "test.jpg")
        
        # Should run without errors
        evaluate(
            model_path=model_path,
            test_data_dir=test_dir,
            batch_size=2,
            device="cpu"
        )
    
    def test_evaluate_with_empty_test_dir(self, simple_model_checkpoint, tmp_path):
        """Test that evaluate handles empty test directory."""
        model_path, classes = simple_model_checkpoint
        
        test_dir = tmp_path / "empty_test"
        test_dir.mkdir()
        
        # Create empty class dirs
        for class_name in classes:
            (test_dir / class_name).mkdir()
        
        # Should handle gracefully
        evaluate(
            model_path=model_path,
            test_data_dir=test_dir,
            batch_size=2,
            device="cpu"
        )


@pytest.mark.skipif(not os.path.exists(_PATH_TEST_DATA), reason="Test data files not found")
class TestEvaluateIntegration:
    """Integration tests for evaluate with real data."""
    
    def test_evaluate_with_real_checkpoint(self, tmp_path):
        """Test evaluate with a real model checkpoint and real data."""
        from car_image_classification_using_cnn.model import create_model
        
        # Create a real checkpoint
        classes = ["Audi", "Hyundai Creta", "Rolls Royce", "Swift", "Tata Safari", "Toyota Innova"]
        model = create_model(num_classes=len(classes))
        
        model_path = tmp_path / "real_model.pth"
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "classes": classes,
            "epoch": 1,
            "val_acc": 50.0,
        }
        torch.save(checkpoint, model_path)
        
        # Use a subset of real test data if available
        if os.path.exists(_PATH_TEST_DATA):
            # This should run without errors
            evaluate(
                model_path=model_path,
                test_data_dir=Path(_PATH_TEST_DATA),
                batch_size=4,
                device="cpu"
            )
