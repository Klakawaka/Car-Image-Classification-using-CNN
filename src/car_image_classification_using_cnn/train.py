from pathlib import Path
from typing import Tuple

import torch
import typer
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from car_image_classification_using_cnn.data import CarImageDataset, get_transforms
from car_image_classification_using_cnn.model import create_model


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> Tuple[float, float]:
    """
    Train the model for one epoch.

    Args:
        model: The model to train.
        train_loader: DataLoader for training data.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device to train on.
        epoch: Current epoch number.

    Returns:
        Tuple of (average_loss, accuracy).
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        pbar.set_postfix({"loss": loss.item(), "acc": 100.0 * correct / total})

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def validate(
    model: nn.Module, val_loader: DataLoader, criterion: nn.Module, device: torch.device, epoch: int
) -> Tuple[float, float]:
    """
    Validate the model.

    Args:
        model: The model to validate.
        val_loader: DataLoader for validation data.
        criterion: Loss function.
        device: Device to validate on.
        epoch: Current epoch number.

    Returns:
        Tuple of (average_loss, accuracy).
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]  ")
    with torch.no_grad():
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Update progress bar
            pbar.set_postfix({"loss": loss.item(), "acc": 100.0 * correct / total})

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def train(
    train_data_path: Path = typer.Option("raw/train", help="Path to training data"),
    test_data_path: Path = typer.Option("raw/test", help="Path to test data"),
    model_type: str = typer.Option("resnet", help="Model type: 'resnet' or 'custom'"),
    pretrained: bool = typer.Option(True, help="Use pretrained weights (ResNet only)"),
    num_epochs: int = typer.Option(10, help="Number of training epochs"),
    batch_size: int = typer.Option(32, help="Batch size for training"),
    learning_rate: float = typer.Option(0.001, help="Learning rate"),
    weight_decay: float = typer.Option(1e-4, help="Weight decay for regularization"),
    output_dir: Path = typer.Option("models", help="Directory to save model checkpoints"),
    device: str = typer.Option("auto", help="Device to train on: 'cpu', 'cuda', 'mps', or 'auto' (auto-detect)"),
) -> None:
    """
    Train the car classification model.

    This function implements a complete training pipeline including:
    - Data loading with augmentation
    - Training and validation loops
    - Model checkpointing
    - Metrics tracking

    Example:
        uv run python -m car_image_classification_using_cnn.train --num-epochs 20 --batch-size 64
    """
    print("=" * 70)
    print("CAR IMAGE CLASSIFICATION - TRAINING")
    print("=" * 70)

    # support diff devices
    if device == "auto":
        if torch.cuda.is_available():
            device_obj = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device_obj = torch.device("mps")
        else:
            device_obj = torch.device("cpu")
    elif device == "cuda":
        if torch.cuda.is_available():
            device_obj = torch.device("cuda")
        else:
            print("Warning: CUDA requested but not available, falling back to CPU")
            device_obj = torch.device("cpu")
    elif device == "mps":
        if torch.backends.mps.is_available():
            device_obj = torch.device("mps")
        else:
            print("Warning: MPS requested but not available, falling back to CPU")
            device_obj = torch.device("cpu")
    else:
        device_obj = torch.device("cpu")
    
    print(f"Using device: {device_obj}")
    if device_obj.type == "mps":
        print("  ✓ Apple Silicon GPU acceleration enabled (MPS)")
    elif device_obj.type == "cuda":
        print("  ✓ NVIDIA GPU acceleration enabled (CUDA)")
    else:
        print(" Using CPU (consider using --device mps on Apple Silicon for faster training)")

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving checkpoints to: {output_dir}")

    # Load datasets
    print("\nLoading datasets...")
    train_dataset = CarImageDataset(
        data_path=Path(train_data_path), transform=get_transforms("train"), image_size=(224, 224)
    )
    test_dataset = CarImageDataset(
        data_path=Path(test_data_path), transform=get_transforms("eval"), image_size=(224, 224)
    )

    print(f"  Train dataset: {len(train_dataset)} images")
    print(f"  Test dataset: {len(test_dataset)} images")
    print(f"  Classes: {train_dataset.classes}")

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    print(f"  Train batches: {len(train_loader)}")
    print(f"  Test batches: {len(test_loader)}")

    # Create model
    print(f"\nCreating {model_type} model...")
    model = create_model(model_type=model_type, num_classes=len(train_dataset.classes), pretrained=pretrained)
    model = model.to(device_obj)

    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {num_params:,}")
    print(f"  Trainable parameters: {num_trainable:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, verbose=True)

    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)

    best_val_acc = 0.0
    best_model_path = output_dir / "best_model.pth"

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        print("-" * 70)

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device_obj, epoch)

        val_loss, val_acc = validate(model, test_loader, criterion, device_obj, epoch)

        scheduler.step(val_loss)

        # Print epoch summary
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                    "train_acc": train_acc,
                    "train_loss": train_loss,
                    "classes": train_dataset.classes,
                },
                best_model_path,
            )
            print(f"  ✓ Saved best model (val_acc: {val_acc:.2f}%)")

        if epoch % 5 == 0:
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch}.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                },
                checkpoint_path,
            )
            print(f"  ✓ Saved checkpoint: {checkpoint_path.name}")

    # Final summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Best model saved to: {best_model_path}")


if __name__ == "__main__":
    typer.run(train)
