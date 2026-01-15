import torch
from torch import nn
from torchvision import models


class CarClassificationCNN(nn.Module):
    """
    CNN model for car image classification using transfer learning with ResNet18.

    This model uses a pre-trained ResNet18 as a backbone and replaces the final
    fully connected layer to classify 6 car brands.
    """

    def __init__(self, num_classes: int = 6, pretrained: bool = True, dropout_rate: float = 0.5) -> None:
        """
        Initialize the car classification model.

        Args:
            num_classes: Number of car classes to predict.
            pretrained: Whether to use pre-trained ImageNet weights.
            dropout_rate: Dropout rate for regularization.
        """
        super().__init__()
        self.num_classes = num_classes

        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)

        num_features = self.backbone.fc.in_features

        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate / 2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224).

        Returns:
            Output logits of shape (batch_size, num_classes).
        """
        return self.backbone(x)

    def freeze_backbone(self) -> None:
        """Freeze the backbone parameters for fine-tuning only the classifier."""
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze the final FC layer
        for param in self.backbone.fc.parameters():
            param.requires_grad = True

    def unfreeze_backbone(self) -> None:
        """Unfreeze all parameters for full fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True


class CustomCNN(nn.Module):
    """
    Custom CNN architecture for car image classification (from scratch).

    This is an alternative to the ResNet-based model, built from scratch.
    """

    def __init__(self, num_classes: int = 6, dropout_rate: float = 0.5) -> None:
        """
        Initialize the custom CNN model.

        Args:
            num_classes: Number of car classes to predict.
            dropout_rate: Dropout rate for regularization.
        """
        super().__init__()
        self.num_classes = num_classes

        # Convolutional layers
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate / 2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224).

        Returns:
            Output logits of shape (batch_size, num_classes).
        """
        x = self.features(x)
        x = self.classifier(x)
        return x


def create_model(model_type: str = "resnet", num_classes: int = 6, pretrained: bool = True) -> nn.Module:
    """
    Factory function to create a model.

    Args:
        model_type: Type of model to create ('resnet' or 'custom').
        num_classes: Number of classes to predict.
        pretrained: Whether to use pre-trained weights (only for 'resnet').

    Returns:
        Initialized model.
    """
    if model_type == "resnet":
        return CarClassificationCNN(num_classes=num_classes, pretrained=pretrained)
    elif model_type == "custom":
        return CustomCNN(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose 'resnet' or 'custom'.")


if __name__ == "__main__":
    print("Testing ResNet-based model...")
    model_resnet = CarClassificationCNN(num_classes=6, pretrained=False)
    x = torch.rand(4, 3, 224, 224)
    output = model_resnet(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Number of parameters: {sum(p.numel() for p in model_resnet.parameters()):,}")

    print("\nTesting Custom CNN model...")
    model_custom = CustomCNN(num_classes=6)
    output = model_custom(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Number of parameters: {sum(p.numel() for p in model_custom.parameters()):,}")
