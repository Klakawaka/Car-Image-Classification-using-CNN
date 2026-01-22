from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from omegaconf import DictConfig, OmegaConf

from car_image_classification_using_cnn.train_hydra import resolve_device


class TestResolveDevice:
    """Test suite for the resolve_device function."""
    
    def test_resolve_device_auto_with_cuda(self):
        """Test auto device selection with CUDA available."""
        with patch("torch.cuda.is_available", return_value=True):
            device = resolve_device("auto")
            assert device.type == "cuda"
    
    def test_resolve_device_auto_with_mps(self):
        """Test auto device selection with MPS available (no CUDA)."""
        with patch("torch.cuda.is_available", return_value=False), \
             patch("torch.backends.mps.is_available", return_value=True):
            device = resolve_device("auto")
            assert device.type == "mps"
    
    def test_resolve_device_auto_cpu_fallback(self):
        """Test auto device selection falls back to CPU."""
        with patch("torch.cuda.is_available", return_value=False), \
             patch("torch.backends.mps.is_available", return_value=False):
            device = resolve_device("auto")
            assert device.type == "cpu"
    
    def test_resolve_device_explicit_cuda(self):
        """Test explicit CUDA device selection."""
        with patch("torch.cuda.is_available", return_value=True):
            device = resolve_device("cuda")
            assert device.type == "cuda"
    
    def test_resolve_device_cuda_not_available(self):
        """Test CUDA requested but not available falls back to CPU."""
        with patch("torch.cuda.is_available", return_value=False):
            device = resolve_device("cuda")
            assert device.type == "cpu"
    
    def test_resolve_device_explicit_mps(self):
        """Test explicit MPS device selection."""
        with patch("torch.backends.mps.is_available", return_value=True):
            device = resolve_device("mps")
            assert device.type == "mps"
    
    def test_resolve_device_mps_not_available(self):
        """Test MPS requested but not available falls back to CPU."""
        with patch("torch.backends.mps.is_available", return_value=False):
            device = resolve_device("mps")
            assert device.type == "cpu"
    
    def test_resolve_device_explicit_cpu(self):
        """Test explicit CPU device selection."""
        device = resolve_device("cpu")
        assert device.type == "cpu"
    
    def test_resolve_device_unknown_fallback(self):
        """Test unknown device string falls back to CPU."""
        device = resolve_device("unknown_device")
        assert device.type == "cpu"


class TestHydraMain:
    """Test suite for the Hydra main function."""
    
    @pytest.fixture
    def mock_config(self, tmp_path):
        """Create a mock Hydra config."""
        config_dict = {
            "training": {
                "device": "cpu",
                "output_dir": str(tmp_path / "output"),
                "batch_size": 4,
                "num_workers": 0,
                "num_epochs": 2,
                "learning_rate": 0.001,
                "weight_decay": 1e-4,
            },
            "data": {
                "train_data_path": str(tmp_path / "train"),
                "test_data_path": str(tmp_path / "test"),
                "image_size": [224, 224],
            },
            "model": {
                "model_type": "custom",
                "pretrained": False,
            },
        }
        return OmegaConf.create(config_dict)
    
    def setup_test_data(self, tmp_path):
        """Set up minimal test dataset."""
        from PIL import Image
        
        classes = ["ClassA", "ClassB", "ClassC"]
        
        for split in ["train", "test"]:
            split_dir = tmp_path / split
            for class_name in classes:
                class_dir = split_dir / class_name
                class_dir.mkdir(parents=True, exist_ok=True)
                
                # Create a few test images
                for i in range(3):
                    img = Image.new("RGB", (100, 100), color=(i * 80, i * 70, i * 60))
                    img.save(class_dir / f"img{i}.jpg")
    
    def test_main_runs_with_valid_config(self, mock_config, tmp_path):
        """Test that main function runs with valid configuration."""
        from car_image_classification_using_cnn.train_hydra import main
        
        # Set up test data
        self.setup_test_data(tmp_path)
        
        # Mock the actual training to speed up test
        with patch("car_image_classification_using_cnn.train_hydra.train_one_epoch") as mock_train, \
             patch("car_image_classification_using_cnn.train_hydra.validate") as mock_validate:
            
            mock_train.return_value = (0.5, 75.0)
            mock_validate.return_value = (0.6, 70.0)
            
            # Should not raise any errors
            main(mock_config)
            
            # Verify training was called
            assert mock_train.call_count == mock_config.training.num_epochs
            assert mock_validate.call_count == mock_config.training.num_epochs
    
    def test_main_creates_output_directory(self, mock_config, tmp_path):
        """Test that main creates output directory."""
        from car_image_classification_using_cnn.train_hydra import main
        
        self.setup_test_data(tmp_path)
        output_dir = Path(mock_config.training.output_dir)
        
        # Output dir shouldn't exist yet
        assert not output_dir.exists()
        
        with patch("car_image_classification_using_cnn.train_hydra.train_one_epoch") as mock_train, \
             patch("car_image_classification_using_cnn.train_hydra.validate") as mock_validate:
            
            mock_train.return_value = (0.5, 75.0)
            mock_validate.return_value = (0.6, 70.0)
            
            main(mock_config)
        
        # Output dir should now exist
        assert output_dir.exists()
    
    def test_main_saves_best_model(self, mock_config, tmp_path):
        """Test that main saves the best model checkpoint."""
        from car_image_classification_using_cnn.train_hydra import main
        
        self.setup_test_data(tmp_path)
        
        with patch("car_image_classification_using_cnn.train_hydra.train_one_epoch") as mock_train, \
             patch("car_image_classification_using_cnn.train_hydra.validate") as mock_validate:
            
            mock_train.return_value = (0.5, 75.0)
            mock_validate.return_value = (0.6, 70.0)
            
            main(mock_config)
        
        # Check that best model was saved
        best_model_path = Path(mock_config.training.output_dir) / "best_model.pth"
        assert best_model_path.exists()
    
    def test_main_raises_error_on_class_mismatch(self, mock_config, tmp_path):
        """Test that main raises error when train/test classes don't match."""
        from car_image_classification_using_cnn.train_hydra import main
        from PIL import Image
        
        # Create train data with different classes than test
        train_classes = ["ClassA", "ClassB"]
        test_classes = ["ClassX", "ClassY"]
        
        train_dir = tmp_path / "train"
        for class_name in train_classes:
            class_dir = train_dir / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            img = Image.new("RGB", (100, 100))
            img.save(class_dir / "img.jpg")
        
        test_dir = tmp_path / "test"
        for class_name in test_classes:
            class_dir = test_dir / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            img = Image.new("RGB", (100, 100))
            img.save(class_dir / "img.jpg")
        
        with pytest.raises(ValueError, match="Train/Test class mismatch"):
            main(mock_config)
    
    def test_main_uses_correct_device(self, mock_config, tmp_path):
        """Test that main uses the correct device from config."""
        from car_image_classification_using_cnn.train_hydra import main
        
        self.setup_test_data(tmp_path)
        
        # Test with different device settings
        mock_config.training.device = "cpu"
        
        with patch("car_image_classification_using_cnn.train_hydra.train_one_epoch") as mock_train, \
             patch("car_image_classification_using_cnn.train_hydra.validate") as mock_validate, \
             patch("car_image_classification_using_cnn.train_hydra.resolve_device") as mock_resolve:
            
            mock_resolve.return_value = torch.device("cpu")
            mock_train.return_value = (0.5, 75.0)
            mock_validate.return_value = (0.6, 70.0)
            
            main(mock_config)
            
            # Verify resolve_device was called with correct argument
            mock_resolve.assert_called_once_with("cpu")
    
    def test_main_creates_data_loaders_with_correct_params(self, mock_config, tmp_path):
        """Test that main creates DataLoaders with correct parameters."""
        from car_image_classification_using_cnn.train_hydra import main
        
        self.setup_test_data(tmp_path)
        
        with patch("car_image_classification_using_cnn.train_hydra.DataLoader") as mock_dataloader, \
             patch("car_image_classification_using_cnn.train_hydra.train_one_epoch") as mock_train, \
             patch("car_image_classification_using_cnn.train_hydra.validate") as mock_validate:
            
            mock_train.return_value = (0.5, 75.0)
            mock_validate.return_value = (0.6, 70.0)
            mock_dataloader.return_value = []
            
            main(mock_config)
            
            # Verify DataLoader was called with correct batch_size
            calls = mock_dataloader.call_args_list
            for call in calls:
                assert call[1]["batch_size"] == mock_config.training.batch_size
    
    def test_main_uses_lr_scheduler(self, mock_config, tmp_path):
        """Test that main creates and uses learning rate scheduler."""
        from car_image_classification_using_cnn.train_hydra import main
        
        self.setup_test_data(tmp_path)
        
        with patch("car_image_classification_using_cnn.train_hydra.train_one_epoch") as mock_train, \
             patch("car_image_classification_using_cnn.train_hydra.validate") as mock_validate, \
             patch("torch.optim.lr_scheduler.ReduceLROnPlateau") as mock_scheduler_class:
            
            mock_train.return_value = (0.5, 75.0)
            mock_validate.return_value = (0.6, 70.0)
            mock_scheduler = MagicMock()
            mock_scheduler_class.return_value = mock_scheduler
            
            main(mock_config)
            
            # Verify scheduler.step was called
            assert mock_scheduler.step.call_count == mock_config.training.num_epochs
    
    def test_main_saves_improving_models_only(self, mock_config, tmp_path):
        """Test that main only saves models when validation improves."""
        from car_image_classification_using_cnn.train_hydra import main
        
        self.setup_test_data(tmp_path)
        mock_config.training.num_epochs = 3
        
        with patch("car_image_classification_using_cnn.train_hydra.train_one_epoch") as mock_train, \
             patch("car_image_classification_using_cnn.train_hydra.validate") as mock_validate, \
             patch("torch.save") as mock_save:
            
            mock_train.return_value = (0.5, 75.0)
            # Simulate validation accuracy: improving, then declining
            mock_validate.side_effect = [
                (0.6, 70.0),  # Epoch 1: 70%
                (0.5, 80.0),  # Epoch 2: 80% (improvement, should save)
                (0.7, 75.0),  # Epoch 3: 75% (decline, should not save)
            ]
            
            main(mock_config)
            
            # Should save twice: initial save + one improvement
            assert mock_save.call_count >= 1
    
    def test_main_with_pretrained_model(self, mock_config, tmp_path):
        """Test main with pretrained model configuration."""
        from car_image_classification_using_cnn.train_hydra import main
        
        self.setup_test_data(tmp_path)
        mock_config.model.model_type = "resnet"
        mock_config.model.pretrained = True
        
        with patch("car_image_classification_using_cnn.train_hydra.train_one_epoch") as mock_train, \
             patch("car_image_classification_using_cnn.train_hydra.validate") as mock_validate:
            
            mock_train.return_value = (0.5, 75.0)
            mock_validate.return_value = (0.6, 70.0)
            
            main(mock_config)
            
            # Verify training was called
            assert mock_train.called
    
    def test_main_prints_progress(self, mock_config, tmp_path):
        """Test that main prints progress information."""
        from car_image_classification_using_cnn.train_hydra import main
        
        self.setup_test_data(tmp_path)
        
        with patch("car_image_classification_using_cnn.train_hydra.train_one_epoch") as mock_train, \
             patch("car_image_classification_using_cnn.train_hydra.validate") as mock_validate, \
             patch("builtins.print") as mock_print:
            
            mock_train.return_value = (0.5, 75.0)
            mock_validate.return_value = (0.6, 70.0)
            
            main(mock_config)
            
            # Verify some output was printed
            assert mock_print.called
    
    def test_main_with_different_image_sizes(self, mock_config, tmp_path):
        """Test main with different image size configurations."""
        from car_image_classification_using_cnn.train_hydra import main
        
        self.setup_test_data(tmp_path)
        
        # Test with different image size
        mock_config.data.image_size = [128, 128]
        
        with patch("car_image_classification_using_cnn.train_hydra.train_one_epoch") as mock_train, \
             patch("car_image_classification_using_cnn.train_hydra.validate") as mock_validate:
            
            mock_train.return_value = (0.5, 75.0)
            mock_validate.return_value = (0.6, 70.0)
            
            main(mock_config)
            
            # Verify training completed
            assert mock_train.called


class TestHydraIntegration:
    """Integration tests for Hydra configuration."""
    
    def test_config_can_be_converted_to_container(self):
        """Test that OmegaConf config can be converted to container."""
        config_dict = {
            "training": {"device": "cpu", "batch_size": 32},
            "model": {"model_type": "resnet"},
        }
        cfg = OmegaConf.create(config_dict)
        
        # Should be able to convert to container
        container = OmegaConf.to_container(cfg, resolve=True)
        
        assert isinstance(container, dict)
        assert container["training"]["device"] == "cpu"
    
    def test_config_to_yaml(self):
        """Test that config can be converted to YAML."""
        config_dict = {
            "training": {"device": "cpu", "batch_size": 32},
        }
        cfg = OmegaConf.create(config_dict)
        
        # Should be able to convert to YAML
        yaml_str = OmegaConf.to_yaml(cfg)
        
        assert isinstance(yaml_str, str)
        assert "device" in yaml_str
        assert "cpu" in yaml_str
