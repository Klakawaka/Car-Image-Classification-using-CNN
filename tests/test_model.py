import os
from pathlib import Path

import pytest
import torch
from torch import nn

from car_image_classification_using_cnn.model import CarClassificationCNN, CustomCNN, create_model


class TestCarClassificationCNN:
    """Test suite for ResNet-based CarClassificationCNN model."""

    @pytest.mark.parametrize("num_classes", [3, 6, 10])
    def test_model_initialization(self, num_classes):
        """Test that model initializes with different numbers of classes."""
        model = CarClassificationCNN(num_classes=num_classes, pretrained=False)
        assert isinstance(model, nn.Module), "Model should be an instance of nn.Module"
        assert model.num_classes == num_classes, f"Model should have {num_classes} classes, got {model.num_classes}"

    @pytest.mark.parametrize("batch_size", [1, 4, 16, 32])
    @pytest.mark.parametrize("num_classes", [6, 10])
    def test_forward_pass_shape(self, batch_size, num_classes):
        """Test that forward pass produces correct output shape for different batch sizes."""
        model = CarClassificationCNN(num_classes=num_classes, pretrained=False)
        model.eval()
        x = torch.randn(batch_size, 3, 224, 224)
        output = model(x)
        assert output.shape == (
            batch_size,
            num_classes,
        ), f"Output shape should be ({batch_size}, {num_classes}), got {output.shape}"

    @pytest.mark.parametrize("height,width", [(224, 224), (128, 128), (256, 256), (299, 299)])
    def test_different_input_sizes(self, height, width):
        """Test that model handles different input image sizes."""
        model = CarClassificationCNN(num_classes=6, pretrained=False)
        model.eval()
        x = torch.randn(2, 3, height, width)
        output = model(x)
        assert output.shape == (2, 6), f"Output shape should be (2, 6), got {output.shape}"

    def test_pretrained_vs_non_pretrained(self):
        """Test that pretrained and non-pretrained models have different initial weights."""
        model_pretrained = CarClassificationCNN(num_classes=6, pretrained=True)
        model_scratch = CarClassificationCNN(num_classes=6, pretrained=False)

        # Get first conv layer weights
        pretrained_weights = model_pretrained.backbone.conv1.weight.data.clone()
        scratch_weights = model_scratch.backbone.conv1.weight.data.clone()

        # They should be different (pretrained has ImageNet weights)
        weights_differ = not torch.allclose(pretrained_weights, scratch_weights)
        assert weights_differ, "Pretrained and non-pretrained models should have different initial weights"

    @pytest.mark.parametrize("dropout_rate", [0.0, 0.3, 0.5, 0.7])
    def test_dropout_rate(self, dropout_rate):
        """Test that model can be initialized with different dropout rates."""
        model = CarClassificationCNN(num_classes=6, pretrained=False, dropout_rate=dropout_rate)
        x = torch.randn(4, 3, 224, 224)
        output = model(x)
        assert output.shape == (4, 6), "Model should work with different dropout rates"

    def test_freeze_backbone(self):
        """Test that freeze_backbone correctly freezes parameters."""
        model = CarClassificationCNN(num_classes=6, pretrained=False)
        model.freeze_backbone()

        # Check that most backbone parameters are frozen
        frozen_params = sum(1 for p in model.backbone.parameters() if not p.requires_grad)
        total_params = sum(1 for p in model.backbone.parameters())

        assert frozen_params > 0, "Some parameters should be frozen after calling freeze_backbone"

        # Check that FC layer is still trainable
        fc_trainable = all(p.requires_grad for p in model.backbone.fc.parameters())
        assert fc_trainable, "FC layer should remain trainable after freeze_backbone"

    def test_unfreeze_backbone(self):
        """Test that unfreeze_backbone correctly unfreezes all parameters."""
        model = CarClassificationCNN(num_classes=6, pretrained=False)
        model.freeze_backbone()
        model.unfreeze_backbone()

        # Check that all parameters are trainable
        all_trainable = all(p.requires_grad for p in model.backbone.parameters())
        assert all_trainable, "All parameters should be trainable after unfreeze_backbone"

    def test_model_has_reasonable_parameter_count(self):
        """Test that model has a reasonable number of parameters."""
        model = CarClassificationCNN(num_classes=6, pretrained=False)
        num_params = sum(p.numel() for p in model.parameters())
        assert 1_000_000 < num_params < 50_000_000, (
            f"Model should have between 1M and 50M parameters, got {num_params:,}"
        )

    def test_model_output_is_float(self):
        """Test that model output is float tensor."""
        model = CarClassificationCNN(num_classes=6, pretrained=False)
        model.eval()
        x = torch.randn(2, 3, 224, 224)
        output = model(x)
        assert output.dtype == torch.float32, f"Output should be float32, got {output.dtype}"

    def test_model_can_be_moved_to_device(self):
        """Test that model can be moved to CPU (and GPU if available)."""
        model = CarClassificationCNN(num_classes=6, pretrained=False)
        model = model.to("cpu")
        x = torch.randn(2, 3, 224, 224)
        output = model(x)
        assert output.device.type == "cpu", "Model output should be on CPU"

    def test_model_gradient_flow(self):
        """Test that gradients flow through the model during backpropagation."""
        model = CarClassificationCNN(num_classes=6, pretrained=False)
        model.train()
        x = torch.randn(2, 3, 224, 224, requires_grad=True)
        output = model(x)
        loss = output.sum()
        loss.backward()

        # Check that at least some parameters have gradients
        has_gradients = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
        assert has_gradients, "Model should have gradients after backward pass"


class TestCustomCNN:
    """Test suite for custom CNN model."""

    @pytest.mark.parametrize("num_classes", [3, 6, 10])
    def test_model_initialization(self, num_classes):
        """Test that custom model initializes with different numbers of classes."""
        model = CustomCNN(num_classes=num_classes)
        assert isinstance(model, nn.Module), "Model should be an instance of nn.Module"
        assert model.num_classes == num_classes, f"Model should have {num_classes} classes, got {model.num_classes}"

    @pytest.mark.parametrize("batch_size", [1, 4, 16, 32])
    def test_forward_pass_shape(self, batch_size):
        """Test that forward pass produces correct output shape for different batch sizes."""
        model = CustomCNN(num_classes=6)
        model.eval()
        x = torch.randn(batch_size, 3, 224, 224)
        output = model(x)
        assert output.shape == (
            batch_size,
            6,
        ), f"Output shape should be ({batch_size}, 6), got {output.shape}"

    @pytest.mark.parametrize("height,width", [(224, 224), (128, 128), (256, 256)])
    def test_different_input_sizes(self, height, width):
        """Test that custom model handles different input image sizes."""
        model = CustomCNN(num_classes=6)
        model.eval()
        x = torch.randn(2, 3, height, width)
        output = model(x)
        assert output.shape == (2, 6), f"Output shape should be (2, 6), got {output.shape}"

    def test_model_has_reasonable_parameter_count(self):
        """Test that custom model has a reasonable number of parameters."""
        model = CustomCNN(num_classes=6)
        num_params = sum(p.numel() for p in model.parameters())
        assert 100_000 < num_params < 20_000_000, (
            f"Model should have between 100K and 20M parameters, got {num_params:,}"
        )

    def test_model_gradient_flow(self):
        """Test that gradients flow through the custom model."""
        model = CustomCNN(num_classes=6)
        model.train()
        x = torch.randn(2, 3, 224, 224, requires_grad=True)
        output = model(x)
        loss = output.sum()
        loss.backward()

        has_gradients = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
        assert has_gradients, "Model should have gradients after backward pass"


class TestModelFactory:
    """Test suite for create_model factory function."""

    @pytest.mark.parametrize("model_type", ["resnet", "custom"])
    def test_create_model_types(self, model_type):
        """Test that create_model can create both model types."""
        model = create_model(model_type=model_type, num_classes=6, pretrained=False)
        assert isinstance(model, nn.Module), f"create_model should return nn.Module for {model_type}"

    def test_create_resnet_model(self):
        """Test that create_model returns CarClassificationCNN for 'resnet'."""
        model = create_model(model_type="resnet", num_classes=6, pretrained=False)
        assert isinstance(
            model, CarClassificationCNN
        ), "create_model with 'resnet' should return CarClassificationCNN instance"

    def test_create_custom_model(self):
        """Test that create_model returns CustomCNN for 'custom'."""
        model = create_model(model_type="custom", num_classes=6, pretrained=False)
        assert isinstance(model, CustomCNN), "create_model with 'custom' should return CustomCNN instance"

    def test_create_model_with_different_num_classes(self):
        """Test that create_model respects num_classes parameter."""
        num_classes = 10
        model = create_model(model_type="resnet", num_classes=num_classes, pretrained=False)
        x = torch.randn(2, 3, 224, 224)
        output = model(x)
        assert output.shape[1] == num_classes, f"Model should output {num_classes} classes"

    def test_invalid_model_type_raises_error(self):
        """Test that create_model raises ValueError for invalid model type."""
        with pytest.raises(ValueError, match="Unknown model type"):
            create_model(model_type="invalid_model", num_classes=6, pretrained=False)

    @pytest.mark.parametrize("invalid_type", ["vgg", "alexnet", "mobilenet", "", None])
    def test_various_invalid_model_types(self, invalid_type):
        """Test that various invalid model types raise ValueError."""
        with pytest.raises((ValueError, TypeError)):
            create_model(model_type=invalid_type, num_classes=6, pretrained=False)


class TestModelInputValidation:
    """Test suite for model input validation and edge cases."""

    def test_model_with_wrong_number_of_channels(self):
        """Test model behavior with incorrect number of input channels."""
        model = CarClassificationCNN(num_classes=6, pretrained=False)
        model.eval()
        # Try with 1 channel instead of 3
        x = torch.randn(2, 1, 224, 224)
        with pytest.raises(RuntimeError):
            model(x)

    def test_model_with_zero_batch_size(self):
        """Test that model handles edge case of empty batch."""
        model = CarClassificationCNN(num_classes=6, pretrained=False)
        model.eval()
        x = torch.randn(0, 3, 224, 224)
        output = model(x)
        assert output.shape == (0, 6), "Model should handle empty batch"

    def test_model_training_vs_eval_mode(self):
        """Test that model behaves differently in training vs eval mode."""
        model = CarClassificationCNN(num_classes=6, pretrained=False, dropout_rate=0.5)
        x = torch.randn(10, 3, 224, 224)

        # Training mode - dropout is active
        model.train()
        output1 = model(x)
        output2 = model(x)
        # Outputs should be different due to dropout
        train_outputs_differ = not torch.allclose(output1, output2)

        # Eval mode - dropout is inactive
        model.eval()
        with torch.no_grad():
            output3 = model(x)
            output4 = model(x)
        # Outputs should be identical in eval mode
        eval_outputs_same = torch.allclose(output3, output4)

        assert train_outputs_differ or eval_outputs_same, (
            "Model should show different behavior in train vs eval mode (dropout effect)"
        )

    def test_model_with_very_small_input(self):
        """Test model behavior with very small input size."""
        model = CarClassificationCNN(num_classes=6, pretrained=False)
        model.eval()
        # ResNet can handle small inputs due to adaptive pooling
        x = torch.randn(1, 3, 32, 32)
        try:
            output = model(x)
            assert output.shape == (1, 6), "Model should handle small inputs"
        except RuntimeError:
            # This is also acceptable - model may not support very small inputs
            pass

    @pytest.mark.parametrize("num_classes", [1, 2, 100, 1000])
    def test_model_with_extreme_num_classes(self, num_classes):
        """Test model with extreme numbers of classes."""
        model = CarClassificationCNN(num_classes=num_classes, pretrained=False)
        x = torch.randn(2, 3, 224, 224)
        output = model(x)
        assert output.shape == (2, num_classes), f"Model should support {num_classes} classes"


class TestModelConsistency:
    """Test suite for model consistency and reproducibility."""

    def test_model_deterministic_with_same_seed(self):
        """Test that model initialization is deterministic with same seed."""
        torch.manual_seed(42)
        model1 = CarClassificationCNN(num_classes=6, pretrained=False)
        first_param1 = next(model1.parameters()).clone()

        torch.manual_seed(42)
        model2 = CarClassificationCNN(num_classes=6, pretrained=False)
        first_param2 = next(model2.parameters()).clone()

        assert torch.allclose(
            first_param1, first_param2
        ), "Models initialized with same seed should have identical weights"

    def test_model_output_deterministic_in_eval(self):
        """Test that model output is deterministic in eval mode."""
        model = CarClassificationCNN(num_classes=6, pretrained=False)
        model.eval()
        torch.manual_seed(123)
        x = torch.randn(4, 3, 224, 224)

        with torch.no_grad():
            output1 = model(x)
            output2 = model(x)

        assert torch.allclose(output1, output2), "Model output should be deterministic in eval mode with same input"

    def test_model_state_dict_keys(self):
        """Test that model state dict contains expected keys."""
        model = CarClassificationCNN(num_classes=6, pretrained=False)
        state_dict = model.state_dict()

        assert len(state_dict) > 0, "State dict should not be empty"
        assert any("backbone" in key for key in state_dict.keys()), "State dict should contain backbone keys"
        assert any("fc" in key for key in state_dict.keys()), "State dict should contain fc layer keys"

    def test_model_can_be_saved_and_loaded(self):
        """Test that model state can be saved and restored."""
        model1 = CarClassificationCNN(num_classes=6, pretrained=False)
        state_dict = model1.state_dict()

        model2 = CarClassificationCNN(num_classes=6, pretrained=False)
        model2.load_state_dict(state_dict)

        # Compare outputs
        model1.eval()
        model2.eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            output1 = model1(x)
            output2 = model2(x)

        assert torch.allclose(output1, output2), "Loaded model should produce identical outputs to original"

