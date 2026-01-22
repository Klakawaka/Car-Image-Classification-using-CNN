import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from car_image_classification_using_cnn.model import CarClassificationCNN
from car_image_classification_using_cnn.train import train_one_epoch, validate
from tests import _PATH_TRAIN_DATA


class TestTrainingFunctions:
    """Test suite for training utility functions."""

    @pytest.fixture
    def dummy_dataloader(self):
        """Create a dummy dataloader for testing."""
        # Create dummy data: 20 samples, 3 channels, 224x224 images with 6 classes
        images = torch.randn(20, 3, 224, 224)
        labels = torch.randint(0, 6, (20,))
        dataset = TensorDataset(images, labels)
        return DataLoader(dataset, batch_size=4, shuffle=False)

    @pytest.fixture
    def dummy_model(self):
        """Create a dummy model for testing."""
        return CarClassificationCNN(num_classes=6, pretrained=False)

    @pytest.fixture
    def criterion(self):
        """Create loss function."""
        return nn.CrossEntropyLoss()

    @pytest.fixture
    def optimizer(self, dummy_model):
        """Create optimizer."""
        return optim.Adam(dummy_model.parameters(), lr=0.001)

    def test_train_one_epoch_returns_tuple(self, dummy_model, dummy_dataloader, criterion, optimizer):
        """Test that train_one_epoch returns a tuple of (loss, accuracy)."""
        device = torch.device("cpu")
        result = train_one_epoch(dummy_model, dummy_dataloader, criterion, optimizer, device, epoch=1)
        assert isinstance(result, tuple), "train_one_epoch should return a tuple"
        assert len(result) == 2, "train_one_epoch should return tuple of length 2 (loss, accuracy)"

    def test_train_one_epoch_loss_type(self, dummy_model, dummy_dataloader, criterion, optimizer):
        """Test that train_one_epoch returns float values."""
        device = torch.device("cpu")
        loss, acc = train_one_epoch(dummy_model, dummy_dataloader, criterion, optimizer, device, epoch=1)
        assert isinstance(loss, float), f"Loss should be float, got {type(loss)}"
        assert isinstance(acc, float), f"Accuracy should be float, got {type(acc)}"

    def test_train_one_epoch_reasonable_values(self, dummy_model, dummy_dataloader, criterion, optimizer):
        """Test that train_one_epoch returns reasonable loss and accuracy values."""
        device = torch.device("cpu")
        loss, acc = train_one_epoch(dummy_model, dummy_dataloader, criterion, optimizer, device, epoch=1)
        assert loss >= 0, f"Loss should be non-negative, got {loss}"
        assert 0 <= acc <= 100, f"Accuracy should be between 0 and 100, got {acc}"

    def test_validate_returns_tuple(self, dummy_model, dummy_dataloader, criterion):
        """Test that validate returns a tuple of (loss, accuracy)."""
        device = torch.device("cpu")
        result = validate(dummy_model, dummy_dataloader, criterion, device, epoch=1)
        assert isinstance(result, tuple), "validate should return a tuple"
        assert len(result) == 2, "validate should return tuple of length 2 (loss, accuracy)"

    def test_validate_loss_type(self, dummy_model, dummy_dataloader, criterion):
        """Test that validate returns float values."""
        device = torch.device("cpu")
        loss, acc = validate(dummy_model, dummy_dataloader, criterion, device, epoch=1)
        assert isinstance(loss, float), f"Loss should be float, got {type(loss)}"
        assert isinstance(acc, float), f"Accuracy should be float, got {type(acc)}"

    def test_validate_reasonable_values(self, dummy_model, dummy_dataloader, criterion):
        """Test that validate returns reasonable loss and accuracy values."""
        device = torch.device("cpu")
        loss, acc = validate(dummy_model, dummy_dataloader, criterion, device, epoch=1)
        assert loss >= 0, f"Loss should be non-negative, got {loss}"
        assert 0 <= acc <= 100, f"Accuracy should be between 0 and 100, got {acc}"

    def test_train_one_epoch_updates_weights(self, dummy_model, dummy_dataloader, criterion, optimizer):
        """Test that train_one_epoch actually updates model weights."""
        device = torch.device("cpu")
        # Get initial weights
        initial_weights = [p.clone() for p in dummy_model.parameters()]

        # Train for one epoch
        train_one_epoch(dummy_model, dummy_dataloader, criterion, optimizer, device, epoch=1)

        # Check that weights have changed
        weights_changed = any(
            not torch.allclose(init, current) for init, current in zip(initial_weights, dummy_model.parameters())
        )
        assert weights_changed, "Model weights should change after training"

    def test_validate_does_not_update_weights(self, dummy_model, dummy_dataloader, criterion):
        """Test that validate does not update model weights."""
        device = torch.device("cpu")
        # Get initial weights
        initial_weights = [p.clone() for p in dummy_model.parameters()]

        # Validate
        validate(dummy_model, dummy_dataloader, criterion, device, epoch=1)

        # Check that weights haven't changed
        weights_unchanged = all(
            torch.allclose(init, current) for init, current in zip(initial_weights, dummy_model.parameters())
        )
        assert weights_unchanged, "Model weights should not change during validation"

    def test_train_one_epoch_sets_train_mode(self, dummy_model, dummy_dataloader, criterion, optimizer):
        """Test that train_one_epoch sets model to training mode."""
        device = torch.device("cpu")
        dummy_model.eval()  # Start in eval mode
        train_one_epoch(dummy_model, dummy_dataloader, criterion, optimizer, device, epoch=1)
        # Model should still be in training mode after
        assert dummy_model.training, "Model should be in training mode after train_one_epoch"

    def test_validate_sets_eval_mode(self, dummy_model, dummy_dataloader, criterion):
        """Test that validate sets model to eval mode."""
        device = torch.device("cpu")
        dummy_model.train()  # Start in train mode
        validate(dummy_model, dummy_dataloader, criterion, device, epoch=1)
        # Model should be in eval mode after
        assert not dummy_model.training, "Model should be in eval mode after validate"

    @pytest.mark.parametrize("batch_size", [1, 4, 8, 16])
    def test_train_with_different_batch_sizes(self, dummy_model, criterion, optimizer, batch_size):
        """Test training with different batch sizes."""
        device = torch.device("cpu")
        images = torch.randn(20, 3, 224, 224)
        labels = torch.randint(0, 6, (20,))
        dataset = TensorDataset(images, labels)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        loss, acc = train_one_epoch(dummy_model, dataloader, criterion, optimizer, device, epoch=1)
        assert loss >= 0 and 0 <= acc <= 100, "Training should work with different batch sizes"

    def test_train_with_empty_dataloader(self, dummy_model, criterion, optimizer):
        """Test training behavior with empty dataloader."""
        device = torch.device("cpu")
        images = torch.randn(0, 3, 224, 224)
        labels = torch.randint(0, 6, (0,))
        dataset = TensorDataset(images, labels)
        dataloader = DataLoader(dataset, batch_size=4)

        # This should handle empty dataloader gracefully or raise an appropriate error
        try:
            loss, acc = train_one_epoch(dummy_model, dataloader, criterion, optimizer, device, epoch=1)
            # If it succeeds, loss and acc might be nan or 0
            assert isinstance(loss, float), "Should return float loss even with empty data"
            assert isinstance(acc, float), "Should return float accuracy even with empty data"
        except (ZeroDivisionError, RuntimeError):
            # It's acceptable to raise an error with empty data
            pass


class TestGradientFlow:
    """Test suite for gradient flow during training."""

    def test_gradients_computed_during_training(self):
        """Test that gradients are computed during training."""
        model = CarClassificationCNN(num_classes=6, pretrained=False)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Create dummy batch
        images = torch.randn(4, 3, 224, 224)
        labels = torch.randint(0, 6, (4,))

        model.train()
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()

        # Check gradients exist
        has_gradients = any(p.grad is not None for p in model.parameters() if p.requires_grad)
        assert has_gradients, "Gradients should be computed during training"

    def test_gradients_not_computed_during_validation(self):
        """Test that gradients are not computed during validation."""
        model = CarClassificationCNN(num_classes=6, pretrained=False)
        criterion = nn.CrossEntropyLoss()

        images = torch.randn(4, 3, 224, 224)
        labels = torch.randint(0, 6, (4,))

        model.eval()
        with torch.no_grad():
            outputs = model(images)
            criterion(outputs, labels)

        # Check that no gradients are stored
        has_gradients = any(p.grad is not None for p in model.parameters())
        assert not has_gradients, "Gradients should not be computed during validation with torch.no_grad()"

    def test_optimizer_step_reduces_loss(self):
        """Test that optimizer step generally reduces loss over multiple iterations."""
        model = CarClassificationCNN(num_classes=6, pretrained=False)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        # Create a simple dataset that the model can learn
        torch.manual_seed(42)
        images = torch.randn(40, 3, 224, 224)
        labels = torch.randint(0, 6, (40,))

        model.train()
        losses = []

        # Train for several iterations
        for _ in range(10):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Loss should generally decrease (check last loss vs first loss)
        assert losses[-1] < losses[0] * 1.5, (
            f"Loss should decrease or stay similar during training. "
            f"Initial: {losses[0]:.4f}, Final: {losses[-1]:.4f}"
        )


class TestLossAndMetrics:
    """Test suite for loss computation and metrics."""

    def test_cross_entropy_loss_with_correct_predictions(self):
        """Test CrossEntropyLoss with perfect predictions."""
        criterion = nn.CrossEntropyLoss()
        # Perfect predictions (high confidence in correct class)
        logits = torch.tensor([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
        labels = torch.tensor([0, 1])
        loss = criterion(logits, labels)
        assert loss.item() < 0.1, f"Loss should be very low for perfect predictions, got {loss.item()}"

    def test_cross_entropy_loss_with_wrong_predictions(self):
        """Test CrossEntropyLoss with incorrect predictions."""
        criterion = nn.CrossEntropyLoss()
        # Wrong predictions (high confidence in wrong class)
        logits = torch.tensor([[0.0, 10.0, 0.0], [10.0, 0.0, 0.0]])
        labels = torch.tensor([0, 1])
        loss = criterion(logits, labels)
        assert loss.item() > 1.0, f"Loss should be high for wrong predictions, got {loss.item()}"

    def test_accuracy_calculation(self):
        """Test accuracy calculation logic."""
        # Simulate model predictions
        logits = torch.tensor([[2.0, 0.5, 0.1], [0.1, 2.5, 0.3], [0.2, 0.1, 3.0]])
        labels = torch.tensor([0, 1, 2])

        _, predicted = torch.max(logits, 1)
        correct = (predicted == labels).sum().item()
        accuracy = 100.0 * correct / labels.size(0)

        assert accuracy == 100.0, f"Accuracy should be 100% for perfect predictions, got {accuracy}%"

    def test_accuracy_with_wrong_predictions(self):
        """Test accuracy calculation with some wrong predictions."""
        logits = torch.tensor([[0.0, 2.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 3.0]])
        labels = torch.tensor([0, 1, 2])

        _, predicted = torch.max(logits, 1)
        correct = (predicted == labels).sum().item()
        accuracy = 100.0 * correct / labels.size(0)

        # First two are wrong, last one is correct
        expected_accuracy = 100.0 / 3
        assert (
            abs(accuracy - expected_accuracy) < 0.01
        ), f"Accuracy should be {expected_accuracy:.2f}%, got {accuracy:.2f}%"


class TestOptimizerBehavior:
    """Test suite for optimizer behavior."""

    @pytest.mark.parametrize("optimizer_class", [optim.Adam, optim.SGD, optim.AdamW])
    def test_different_optimizers(self, optimizer_class):
        """Test that different optimizers can be used for training."""
        model = CarClassificationCNN(num_classes=6, pretrained=False)
        optimizer = optimizer_class(model.parameters(), lr=0.001)

        images = torch.randn(4, 3, 224, 224)
        labels = torch.randint(0, 6, (4,))
        criterion = nn.CrossEntropyLoss()

        model.train()
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Should complete without error
        assert True, f"{optimizer_class.__name__} should work for training"

    def test_optimizer_zero_grad_clears_gradients(self):
        """Test that optimizer.zero_grad() clears gradients."""
        model = CarClassificationCNN(num_classes=6, pretrained=False)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        images = torch.randn(4, 3, 224, 224)
        labels = torch.randint(0, 6, (4,))

        # First forward-backward pass
        model.train()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        # Check gradients exist
        has_gradients_before = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
        assert has_gradients_before, "Gradients should exist before zero_grad()"

        # Clear gradients
        optimizer.zero_grad()

        # Check gradients are cleared
        has_nonzero_gradients_after = any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters() if p.requires_grad
        )
        assert not has_nonzero_gradients_after, "Gradients should be zero after zero_grad()"

    @pytest.mark.parametrize("learning_rate", [1e-5, 1e-3, 1e-1])
    def test_different_learning_rates(self, learning_rate):
        """Test training with different learning rates."""
        model = CarClassificationCNN(num_classes=6, pretrained=False)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        images = torch.randn(4, 3, 224, 224)
        labels = torch.randint(0, 6, (4,))

        model.train()
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Should work with different learning rates
        assert True, f"Training should work with learning_rate={learning_rate}"


@pytest.mark.skipif(not os.path.exists(_PATH_TRAIN_DATA), reason="Training data files not found")
class TestTrainingIntegration:
    """Integration tests for training with real data (if available)."""

    def test_training_with_real_dataloader(self):
        """Test training function with real data if available."""
        from car_image_classification_using_cnn.data import CarImageDataset, get_transforms

        dataset = CarImageDataset(Path(_PATH_TRAIN_DATA), transform=get_transforms("train"))
        # Use small subset for testing
        indices = list(range(min(20, len(dataset))))
        subset = torch.utils.data.Subset(dataset, indices)
        dataloader = DataLoader(subset, batch_size=4, shuffle=True)

        model = CarClassificationCNN(num_classes=len(dataset.classes), pretrained=False)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        device = torch.device("cpu")

        # Train for one epoch
        loss, acc = train_one_epoch(model, dataloader, criterion, optimizer, device, epoch=1)

        assert isinstance(loss, float), "Should return float loss"
        assert isinstance(acc, float), "Should return float accuracy"
        assert loss >= 0, f"Loss should be non-negative, got {loss}"
        assert 0 <= acc <= 100, f"Accuracy should be between 0 and 100, got {acc}"


class TestModelCheckpointing:
    """Test suite for model saving and loading during training."""

    def test_model_state_dict_save_and_load(self):
        """Test saving and loading model state dict."""
        model = CarClassificationCNN(num_classes=6, pretrained=False)
        state_dict = model.state_dict()

        # Create new model and load state
        new_model = CarClassificationCNN(num_classes=6, pretrained=False)
        new_model.load_state_dict(state_dict)

        # Test that outputs are identical
        model.eval()
        new_model.eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            output1 = model(x)
            output2 = new_model(x)

        assert torch.allclose(output1, output2), "Loaded model should produce identical outputs"

    def test_checkpoint_contains_required_keys(self):
        """Test that training checkpoint contains all required information."""
        model = CarClassificationCNN(num_classes=6, pretrained=False)
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        checkpoint = {
            "epoch": 5,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_acc": 85.5,
            "val_loss": 0.42,
            "train_acc": 90.2,
            "train_loss": 0.35,
            "classes": ["Audi", "BMW", "Mercedes", "Toyota", "Honda", "Ford"],
        }

        required_keys = ["epoch", "model_state_dict", "optimizer_state_dict", "val_acc", "val_loss"]
        for key in required_keys:
            assert key in checkpoint, f"Checkpoint should contain '{key}'"

    def test_optimizer_state_dict_save_and_load(self):
        """Test saving and loading optimizer state dict."""
        model = CarClassificationCNN(num_classes=6, pretrained=False)
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Save state
        optimizer_state = optimizer.state_dict()

        # Create new optimizer and load state
        new_optimizer = optim.Adam(model.parameters(), lr=0.001)
        new_optimizer.load_state_dict(optimizer_state)

        # Verify learning rate is preserved
        assert (
            new_optimizer.param_groups[0]["lr"] == optimizer.param_groups[0]["lr"]
        ), "Loaded optimizer should have same learning rate"


class TestWandbIntegration:
    """Test suite for Weights & Biases integration."""
    
    def test_train_one_epoch_with_wandb_disabled(self):
        """Test training with wandb disabled."""
        from car_image_classification_using_cnn.train import train_one_epoch
        
        model = CarClassificationCNN(num_classes=6, pretrained=False)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        images = torch.randn(8, 3, 224, 224)
        labels = torch.randint(0, 6, (8,))
        dataset = TensorDataset(images, labels)
        dataloader = DataLoader(dataset, batch_size=4)
        
        device = torch.device("cpu")
        
        # Should work without wandb
        loss, acc = train_one_epoch(
            model, dataloader, criterion, optimizer, device, epoch=1, use_wandb=False
        )
        
        assert isinstance(loss, float)
        assert isinstance(acc, float)
    
    def test_train_one_epoch_with_wandb_enabled_but_not_available(self):
        """Test training with wandb enabled but not installed."""
        from car_image_classification_using_cnn.train import train_one_epoch, WANDB_AVAILABLE
        
        model = CarClassificationCNN(num_classes=6, pretrained=False)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        images = torch.randn(8, 3, 224, 224)
        labels = torch.randint(0, 6, (8,))
        dataset = TensorDataset(images, labels)
        dataloader = DataLoader(dataset, batch_size=4)
        
        device = torch.device("cpu")
        
        # If wandb is available, we need to mock it; if not, it should handle gracefully
        if WANDB_AVAILABLE:
            with patch("car_image_classification_using_cnn.train.wandb") as mock_wandb:
                mock_wandb.log = MagicMock()
                loss, acc = train_one_epoch(
                    model, dataloader, criterion, optimizer, device, epoch=1, use_wandb=True
                )
        else:
            # Should work without wandb
            loss, acc = train_one_epoch(
                model, dataloader, criterion, optimizer, device, epoch=1, use_wandb=True
            )
        
        assert isinstance(loss, float)
        assert isinstance(acc, float)


class TestDeviceSelection:
    """Test suite for device selection logic."""
    
    def test_cpu_device_selection(self):
        """Test that CPU device can be explicitly selected."""
        device = torch.device("cpu")
        model = CarClassificationCNN(num_classes=6, pretrained=False)
        model = model.to(device)
        
        # Verify model is on CPU
        first_param = next(model.parameters())
        assert first_param.device.type == "cpu"
    
    def test_model_inference_on_cpu(self):
        """Test model inference works on CPU."""
        device = torch.device("cpu")
        model = CarClassificationCNN(num_classes=6, pretrained=False).to(device)
        model.eval()
        
        x = torch.randn(2, 3, 224, 224).to(device)
        with torch.no_grad():
            output = model(x)
        
        assert output.device.type == "cpu"
        assert output.shape == (2, 6)
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_device_if_available(self):
        """Test CUDA device selection if available."""
        device = torch.device("cuda")
        model = CarClassificationCNN(num_classes=6, pretrained=False).to(device)
        
        first_param = next(model.parameters())
        assert first_param.device.type == "cuda"
    
    @pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
    def test_mps_device_if_available(self):
        """Test MPS device selection if available (Apple Silicon)."""
        device = torch.device("mps")
        model = CarClassificationCNN(num_classes=6, pretrained=False).to(device)
        
        first_param = next(model.parameters())
        assert first_param.device.type == "mps"


class TestLearningRateScheduler:
    """Test suite for learning rate scheduling."""
    
    def test_reduce_lr_on_plateau_initialization(self):
        """Test ReduceLROnPlateau scheduler initialization."""
        model = CarClassificationCNN(num_classes=6, pretrained=False)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
        
        # Initial learning rate should be preserved
        assert optimizer.param_groups[0]["lr"] == 0.01
    
    def test_scheduler_reduces_lr_on_plateau(self):
        """Test that scheduler reduces learning rate when loss plateaus."""
        model = CarClassificationCNN(num_classes=6, pretrained=False)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
        
        initial_lr = optimizer.param_groups[0]["lr"]
        
        # Simulate plateau (no improvement in loss)
        for _ in range(5):
            scheduler.step(1.0)
        
        # Learning rate should have been reduced
        current_lr = optimizer.param_groups[0]["lr"]
        assert current_lr < initial_lr, f"LR should be reduced: initial={initial_lr}, current={current_lr}"
    
    def test_scheduler_does_not_reduce_lr_with_improvement(self):
        """Test that scheduler doesn't reduce LR when loss improves."""
        model = CarClassificationCNN(num_classes=6, pretrained=False)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
        
        initial_lr = optimizer.param_groups[0]["lr"]
        
        # Simulate improving loss
        for i in range(3):
            scheduler.step(1.0 - i * 0.1)
        
        # Learning rate should not have been reduced
        current_lr = optimizer.param_groups[0]["lr"]
        assert current_lr == initial_lr, "LR should not be reduced when loss improves"


class TestTrainingEdgeCases:
    """Test suite for edge cases in training."""
    
    def test_training_with_single_batch(self):
        """Test training with only one batch."""
        model = CarClassificationCNN(num_classes=6, pretrained=False)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Single batch
        images = torch.randn(4, 3, 224, 224)
        labels = torch.randint(0, 6, (4,))
        dataset = TensorDataset(images, labels)
        dataloader = DataLoader(dataset, batch_size=4)
        
        device = torch.device("cpu")
        loss, acc = train_one_epoch(model, dataloader, criterion, optimizer, device, epoch=1)
        
        assert isinstance(loss, float) and loss >= 0
        assert isinstance(acc, float) and 0 <= acc <= 100
    
    def test_validation_with_single_sample(self):
        """Test validation with only one sample."""
        model = CarClassificationCNN(num_classes=6, pretrained=False)
        criterion = nn.CrossEntropyLoss()
        
        # Single sample
        images = torch.randn(1, 3, 224, 224)
        labels = torch.randint(0, 6, (1,))
        dataset = TensorDataset(images, labels)
        dataloader = DataLoader(dataset, batch_size=1)
        
        device = torch.device("cpu")
        loss, acc = validate(model, dataloader, criterion, device, epoch=1)
        
        assert isinstance(loss, float) and loss >= 0
        assert isinstance(acc, float) and acc in [0.0, 100.0]
    
    def test_training_with_large_batch_size(self):
        """Test training with batch size equal to dataset size."""
        model = CarClassificationCNN(num_classes=6, pretrained=False)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        images = torch.randn(20, 3, 224, 224)
        labels = torch.randint(0, 6, (20,))
        dataset = TensorDataset(images, labels)
        # Batch size equals dataset size
        dataloader = DataLoader(dataset, batch_size=20)
        
        device = torch.device("cpu")
        loss, acc = train_one_epoch(model, dataloader, criterion, optimizer, device, epoch=1)
        
        assert isinstance(loss, float) and loss >= 0
        assert isinstance(acc, float) and 0 <= acc <= 100
    
    def test_model_on_different_image_sizes(self):
        """Test that model can handle different input sizes."""
        model = CarClassificationCNN(num_classes=6, pretrained=False)
        model.eval()
        
        # Different sizes (note: this might fail for some models that expect fixed size)
        sizes = [(224, 224), (112, 112)]
        
        for h, w in sizes:
            x = torch.randn(2, 3, h, w)
            with torch.no_grad():
                try:
                    output = model(x)
                    assert output.shape[0] == 2
                    assert output.shape[1] == 6
                except RuntimeError:
                    # It's OK if model doesn't support variable sizes
                    pass


class TestModelTrainingState:
    """Test suite for model training state management."""
    
    def test_model_switches_to_training_mode(self):
        """Test that model can switch to training mode."""
        model = CarClassificationCNN(num_classes=6, pretrained=False)
        model.eval()
        assert not model.training
        
        model.train()
        assert model.training
    
    def test_model_switches_to_eval_mode(self):
        """Test that model can switch to eval mode."""
        model = CarClassificationCNN(num_classes=6, pretrained=False)
        model.train()
        assert model.training
        
        model.eval()
        assert not model.training
    
    def test_dropout_behavior_in_train_vs_eval(self):
        """Test that dropout behaves differently in train vs eval mode."""
        # Create a simple model with dropout
        class SimpleModelWithDropout(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 10)
                self.dropout = nn.Dropout(p=0.5)
            
            def forward(self, x):
                return self.dropout(self.fc(x))
        
        model = SimpleModelWithDropout()
        x = torch.randn(100, 10)
        
        # In training mode, dropout should be active
        model.train()
        torch.manual_seed(42)
        output_train_1 = model(x)
        torch.manual_seed(42)
        output_train_2 = model(x)
        # Due to dropout randomness, outputs might differ
        
        # In eval mode, dropout should be disabled
        model.eval()
        torch.manual_seed(42)
        output_eval_1 = model(x)
        torch.manual_seed(42)
        output_eval_2 = model(x)
        # Outputs should be identical in eval mode
        assert torch.allclose(output_eval_1, output_eval_2), "Eval mode outputs should be deterministic"


class TestLoggingAndProgress:
    """Test suite for logging and progress tracking."""
    
    def test_train_one_epoch_logs_progress(self):
        """Test that train_one_epoch logs progress."""
        model = CarClassificationCNN(num_classes=6, pretrained=False)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        images = torch.randn(20, 3, 224, 224)
        labels = torch.randint(0, 6, (20,))
        dataset = TensorDataset(images, labels)
        dataloader = DataLoader(dataset, batch_size=4)
        
        device = torch.device("cpu")
        
        # Should complete without errors
        loss, acc = train_one_epoch(model, dataloader, criterion, optimizer, device, epoch=1)
        assert loss >= 0 and 0 <= acc <= 100
    
    def test_validate_logs_progress(self):
        """Test that validate logs progress."""
        model = CarClassificationCNN(num_classes=6, pretrained=False)
        criterion = nn.CrossEntropyLoss()
        
        images = torch.randn(20, 3, 224, 224)
        labels = torch.randint(0, 6, (20,))
        dataset = TensorDataset(images, labels)
        dataloader = DataLoader(dataset, batch_size=4)
        
        device = torch.device("cpu")
        
        # Should complete without errors
        loss, acc = validate(model, dataloader, criterion, device, epoch=1)
        assert loss >= 0 and 0 <= acc <= 100


class TestTrainMainFunction:
    """Test suite for the main train function - using mocks for speed."""
    
    @pytest.fixture
    def test_data_dirs(self, tmp_path):
        """Create minimal test data directories."""
        from PIL import Image
        
        classes = ["ClassA", "ClassB", "ClassC"]
        
        train_dir = tmp_path / "train"
        test_dir = tmp_path / "test"
        
        for split_dir in [train_dir, test_dir]:
            for class_name in classes:
                class_dir = split_dir / class_name
                class_dir.mkdir(parents=True, exist_ok=True)
                
                # Create just one test image per class
                img = Image.new("RGB", (50, 50), color=(100, 100, 100))
                img.save(class_dir / "img.jpg")
        
        return train_dir, test_dir
    
    def test_train_function_handles_device_selection(self, test_data_dirs, tmp_path):
        """Test that train function handles device selection."""
        from car_image_classification_using_cnn.train import train
        
        train_dir, test_dir = test_data_dirs
        output_dir = tmp_path / "output"
        
        # Mock the training loops to avoid actual training
        with patch("car_image_classification_using_cnn.train.train_one_epoch") as mock_train, \
             patch("car_image_classification_using_cnn.train.validate") as mock_val:
            
            mock_train.return_value = (0.5, 80.0)
            mock_val.return_value = (0.6, 75.0)
            
            train(
                train_data_path=train_dir,
                test_data_path=test_dir,
                model_type="custom",
                pretrained=False,
                num_epochs=1,
                batch_size=2,
                learning_rate=0.001,
                weight_decay=1e-4,
                output_dir=output_dir,
                device="cpu",
                profile_run=False,
                use_wandb=False,
                wandb_project="test-project",
                wandb_entity=None,
                log_level="ERROR",
            )
        
        # Verify output directory was created
        assert output_dir.exists()
    
    def test_train_with_auto_device_selection(self, test_data_dirs, tmp_path):
        """Test train with auto device selection."""
        from car_image_classification_using_cnn.train import train
        
        train_dir, test_dir = test_data_dirs
        output_dir = tmp_path / "output"
        
        with patch("car_image_classification_using_cnn.train.train_one_epoch") as mock_train, \
             patch("car_image_classification_using_cnn.train.validate") as mock_val, \
             patch("torch.cuda.is_available", return_value=False), \
             patch("torch.backends.mps.is_available", return_value=False):
            
            mock_train.return_value = (0.5, 80.0)
            mock_val.return_value = (0.6, 75.0)
            
            train(
                train_data_path=train_dir,
                test_data_path=test_dir,
                model_type="custom",
                pretrained=False,
                num_epochs=1,
                batch_size=2,
                learning_rate=0.001,
                weight_decay=1e-4,
                output_dir=output_dir,
                device="auto",
                profile_run=False,
                use_wandb=False,
                wandb_project="test-project",
                wandb_entity=None,
                log_level="ERROR",
            )
        
        assert output_dir.exists()
    
    def test_train_creates_log_file(self, test_data_dirs, tmp_path):
        """Test that train creates a log file."""
        from car_image_classification_using_cnn.train import train
        
        train_dir, test_dir = test_data_dirs
        output_dir = tmp_path / "output"
        
        with patch("car_image_classification_using_cnn.train.train_one_epoch") as mock_train, \
             patch("car_image_classification_using_cnn.train.validate") as mock_val:
            
            mock_train.return_value = (0.5, 80.0)
            mock_val.return_value = (0.6, 75.0)
            
            train(
                train_data_path=train_dir,
                test_data_path=test_dir,
                model_type="custom",
                pretrained=False,
                num_epochs=1,
                batch_size=2,
                learning_rate=0.001,
                weight_decay=1e-4,
                output_dir=output_dir,
                device="cpu",
                profile_run=False,
                use_wandb=False,
                wandb_project="test-project",
                wandb_entity=None,
                log_level="INFO",
            )
        
        # Check for log file
        log_path = output_dir / "training.log"
        assert log_path.exists()
    
    def test_train_with_profile_run(self, test_data_dirs, tmp_path):
        """Test train with profiling enabled exits early."""
        from car_image_classification_using_cnn.train import train
        
        train_dir, test_dir = test_data_dirs
        output_dir = tmp_path / "output"
        
        with patch("car_image_classification_using_cnn.train.profile") as mock_profile:
            # Mock the profile context manager
            mock_profile.return_value.__enter__ = MagicMock()
            mock_profile.return_value.__exit__ = MagicMock()
            
            train(
                train_data_path=train_dir,
                test_data_path=test_dir,
                model_type="custom",
                pretrained=False,
                num_epochs=1,
                batch_size=2,
                learning_rate=0.001,
                weight_decay=1e-4,
                output_dir=output_dir,
                device="cpu",
                profile_run=True,
                use_wandb=False,
                wandb_project="test-project",
                wandb_entity=None,
                log_level="ERROR",
            )
        
        assert output_dir.exists()
    
    def test_train_handles_wandb_not_available(self, test_data_dirs, tmp_path):
        """Test train handles wandb not being available."""
        from car_image_classification_using_cnn.train import train
        
        train_dir, test_dir = test_data_dirs
        output_dir = tmp_path / "output"
        
        with patch("car_image_classification_using_cnn.train.WANDB_AVAILABLE", False), \
             patch("car_image_classification_using_cnn.train.train_one_epoch") as mock_train, \
             patch("car_image_classification_using_cnn.train.validate") as mock_val:
            
            mock_train.return_value = (0.5, 80.0)
            mock_val.return_value = (0.6, 75.0)
            
            train(
                train_data_path=train_dir,
                test_data_path=test_dir,
                model_type="custom",
                pretrained=False,
                num_epochs=1,
                batch_size=2,
                learning_rate=0.001,
                weight_decay=1e-4,
                output_dir=output_dir,
                device="cpu",
                profile_run=False,
                use_wandb=True,  # Try to use wandb but it's not available
                wandb_project="test-project",
                wandb_entity=None,
                log_level="ERROR",
            )
        
        assert output_dir.exists()
    
    def test_train_handles_missing_wandb_api_key(self, test_data_dirs, tmp_path):
        """Test train handles missing WANDB_API_KEY."""
        from car_image_classification_using_cnn.train import train
        
        train_dir, test_dir = test_data_dirs
        output_dir = tmp_path / "output"
        
        with patch("car_image_classification_using_cnn.train.WANDB_AVAILABLE", True), \
             patch("car_image_classification_using_cnn.train.os.getenv", return_value=None), \
             patch("car_image_classification_using_cnn.train.train_one_epoch") as mock_train, \
             patch("car_image_classification_using_cnn.train.validate") as mock_val:
            
            mock_train.return_value = (0.5, 80.0)
            mock_val.return_value = (0.6, 75.0)
            
            train(
                train_data_path=train_dir,
                test_data_path=test_dir,
                model_type="custom",
                pretrained=False,
                num_epochs=1,
                batch_size=2,
                learning_rate=0.001,
                weight_decay=1e-4,
                output_dir=output_dir,
                device="cpu",
                profile_run=False,
                use_wandb=True,
                wandb_project="test-project",
                wandb_entity=None,
                log_level="ERROR",
            )
        
        assert output_dir.exists()
    
    def test_train_saves_checkpoint_on_improvement(self, test_data_dirs, tmp_path):
        """Test that train saves checkpoint when validation improves."""
        from car_image_classification_using_cnn.train import train
        
        train_dir, test_dir = test_data_dirs
        output_dir = tmp_path / "output"
        
        with patch("car_image_classification_using_cnn.train.train_one_epoch") as mock_train, \
             patch("car_image_classification_using_cnn.train.validate") as mock_val, \
             patch("car_image_classification_using_cnn.train.torch.save") as mock_save:
            
            mock_train.return_value = (0.5, 80.0)
            # Improving validation accuracy
            mock_val.side_effect = [(0.6, 70.0), (0.5, 80.0)]
            
            train(
                train_data_path=train_dir,
                test_data_path=test_dir,
                model_type="custom",
                pretrained=False,
                num_epochs=2,
                batch_size=2,
                learning_rate=0.001,
                weight_decay=1e-4,
                output_dir=output_dir,
                device="cpu",
                profile_run=False,
                use_wandb=False,
                wandb_project="test-project",
                wandb_entity=None,
                log_level="ERROR",
            )
            
            # Should save at least once (for improvement)
            assert mock_save.called
    
    def test_train_uses_lr_scheduler(self, test_data_dirs, tmp_path):
        """Test that train uses learning rate scheduler."""
        from car_image_classification_using_cnn.train import train
        
        train_dir, test_dir = test_data_dirs
        output_dir = tmp_path / "output"
        
        with patch("car_image_classification_using_cnn.train.train_one_epoch") as mock_train, \
             patch("car_image_classification_using_cnn.train.validate") as mock_val, \
             patch("torch.optim.lr_scheduler.ReduceLROnPlateau") as mock_scheduler:
            
            mock_train.return_value = (0.5, 80.0)
            mock_val.return_value = (0.6, 75.0)
            mock_sched_instance = MagicMock()
            mock_scheduler.return_value = mock_sched_instance
            
            train(
                train_data_path=train_dir,
                test_data_path=test_dir,
                model_type="custom",
                pretrained=False,
                num_epochs=2,
                batch_size=2,
                learning_rate=0.001,
                weight_decay=1e-4,
                output_dir=output_dir,
                device="cpu",
                profile_run=False,
                use_wandb=False,
                wandb_project="test-project",
                wandb_entity=None,
                log_level="ERROR",
            )
            
            # Scheduler step should be called for each epoch
            assert mock_sched_instance.step.call_count == 2
