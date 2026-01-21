from pathlib import Path
from typing import Tuple
import os

import torch
import typer
from dotenv import load_dotenv
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from torch.profiler import profile, ProfilerActivity

from car_image_classification_using_cnn.data import CarImageDataset, get_transforms
from car_image_classification_using_cnn.logger import get_logger, setup_logger
from car_image_classification_using_cnn.model import create_model

load_dotenv()

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

logger = get_logger()


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    use_wandb: bool = False,
) -> Tuple[float, float]:
    logger.info(f"Starting training epoch {epoch}")
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    for batch_idx, (images, labels) in enumerate(pbar):
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

        if batch_idx % 10 == 0:
            logger.debug(f"Batch {batch_idx}/{len(train_loader)}: loss={loss.item():.4f}, acc={100.0 * correct / total:.2f}%")
            
            if use_wandb and WANDB_AVAILABLE:
                wandb.log({
                    "batch": epoch * len(train_loader) + batch_idx,
                    "train_loss_step": loss.item(),
                    "train_acc_step": 100.0 * correct / total,
                })

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    logger.info(f"Epoch {epoch} training complete: loss={epoch_loss:.4f}, acc={epoch_acc:.2f}%")
    return epoch_loss, epoch_acc


def validate(
    model: nn.Module, val_loader: DataLoader, criterion: nn.Module, device: torch.device, epoch: int
) -> Tuple[float, float]:
    logger.info(f"Starting validation for epoch {epoch}")
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

            pbar.set_postfix({"loss": loss.item(), "acc": 100.0 * correct / total})

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    logger.info(f"Epoch {epoch} validation complete: loss={epoch_loss:.4f}, acc={epoch_acc:.2f}%")
    return epoch_loss, epoch_acc


@logger.catch
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
    profile_run: bool = typer.Option(False, help="Enable profiling for performance analysis (profiles 30 batches and exits)"),
    use_wandb: bool = typer.Option(False, help="Enable Weights & Biases logging"),
    wandb_project: str = typer.Option("car-classification", help="W&B project name"),
    wandb_entity: str = typer.Option(None, help="W&B entity (team) name"),
    log_level: str = typer.Option("INFO", help="Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL"),
) -> None:
    """
    Train the car classification model.

    This function implements a complete training pipeline including:
    - Data loading with augmentation
    - Training and validation loops
    - Model checkpointing
    - Metrics tracking
    - Weights & Biases experiment tracking (optional)

    Example:
        uv run python -m car_image_classification_using_cnn.train --num-epochs 20 --batch-size 64 --use-wandb
    """
    log_file = Path(output_dir) / "training.log"
    setup_logger(log_file=log_file, level=log_level)

    logger.info("=" * 70)
    logger.info("CAR IMAGE CLASSIFICATION - TRAINING")
    logger.info("=" * 70)

    print("=" * 70)
    print("CAR IMAGE CLASSIFICATION - TRAINING")
    print("=" * 70)

    if use_wandb:
        if not WANDB_AVAILABLE:
            logger.warning("wandb is not installed. Install it with: uv add wandb")
            print("Warning: wandb is not installed. Continuing without W&B logging.")
            use_wandb = False
        else:
            # Get API key from environment
            wandb_api_key = os.getenv("WANDB_API_KEY")
            if not wandb_api_key:
                logger.warning("WANDB_API_KEY not found in environment. Disabling W&B logging.")
                print("Warning: WANDB_API_KEY not found in .env file. Disabling W&B logging.")
                use_wandb = False
            else:
                logger.info("Initializing Weights & Biases")
                wandb.init(
                    project=wandb_project,
                    entity=wandb_entity,
                    config={
                        "model_type": model_type,
                        "pretrained": pretrained,
                        "num_epochs": num_epochs,
                        "batch_size": batch_size,
                        "learning_rate": learning_rate,
                        "weight_decay": weight_decay,
                        "device": device,
                        "train_data_path": str(train_data_path),
                        "test_data_path": str(test_data_path),
                    },
                )
                logger.info(f"W&B Run: {wandb.run.name}")
                print(f"  ✓ Weights & Biases initialized: {wandb.run.name}")

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
            logger.warning("CUDA requested but not available, falling back to CPU")
            print("Warning: CUDA requested but not available, falling back to CPU")
            device_obj = torch.device("cpu")
    elif device == "mps":
        if torch.backends.mps.is_available():
            device_obj = torch.device("mps")
        else:
            logger.warning("MPS requested but not available, falling back to CPU")
            print("Warning: MPS requested but not available, falling back to CPU")
            device_obj = torch.device("cpu")
    else:
        device_obj = torch.device("cpu")

    print(f"Using device: {device_obj}")
    if device_obj.type == "mps":
        logger.info("Apple Silicon GPU acceleration enabled (MPS)")
        print("  ✓ Apple Silicon GPU acceleration enabled (MPS)")
    elif device_obj.type == "cuda":
        logger.info("NVIDIA GPU acceleration enabled (CUDA)")
        print("  ✓ NVIDIA GPU acceleration enabled (CUDA)")
    else:
        logger.info("Using CPU")
        print("  Using CPU (consider using --device mps on Apple Silicon for faster training)")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving checkpoints to: {output_dir}")
    print(f"Saving checkpoints to: {output_dir}")

    logger.info("Loading datasets...")
    print("\nLoading datasets...")
    try:
        train_dataset = CarImageDataset(
            data_path=Path(train_data_path), transform=get_transforms("train"), image_size=(224, 224)
        )
        test_dataset = CarImageDataset(
            data_path=Path(test_data_path), transform=get_transforms("eval"), image_size=(224, 224)
        )
        logger.info(f"Train dataset: {len(train_dataset)} images")
        logger.info(f"Test dataset: {len(test_dataset)} images")
        logger.info(f"Classes: {train_dataset.classes}")
    except Exception as e:
        logger.error(f"Failed to load datasets: {e}")
        raise

    print(f"  Train dataset: {len(train_dataset)} images")
    print(f"  Test dataset: {len(test_dataset)} images")
    print(f"  Classes: {train_dataset.classes}")

    pin = device_obj.type in ("cuda", "mps")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=pin)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=pin)

    logger.debug(f"Train batches: {len(train_loader)}")
    logger.debug(f"Test batches: {len(test_loader)}")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Test batches: {len(test_loader)}")

    logger.info(f"Creating {model_type} model...")
    print(f"\nCreating {model_type} model...")
    model = create_model(model_type=model_type, num_classes=len(train_dataset.classes), pretrained=pretrained)
    model = model.to(device_obj)

    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {num_params:,}")
    logger.info(f"Trainable parameters: {num_trainable:,}")
    print(f"  Total parameters: {num_params:,}")
    print(f"  Trainable parameters: {num_trainable:,}")

    if use_wandb and WANDB_AVAILABLE:
        wandb.watch(model, log="all", log_freq=100)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, verbose=True)

    if profile_run:
        print("\n[Profiling enabled] Profiling 30 training batches...\n")
        logger.info("Starting profiling mode")

        activities = [ProfilerActivity.CPU]
        if device_obj.type == "cuda":
            activities.append(ProfilerActivity.CUDA)

        with profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
        ) as prof:
            model.train()
            for batch_idx, (images, labels) in enumerate(train_loader):
                if batch_idx >= 30:
                    break

                images, labels = images.to(device_obj), labels.to(device_obj)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                prof.step()

        print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=25))
        logger.info("Profiling finished")
        print("\nProfiling finished. Exiting without full training.\n")
        return

    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)
    logger.info("Starting training loop")

    best_val_acc = 0.0
    best_model_path = output_dir / "best_model.pth"

    for epoch in range(1, num_epochs + 1):
        logger.info(f"Epoch {epoch}/{num_epochs}")
        print(f"\nEpoch {epoch}/{num_epochs}")
        print("-" * 70)

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device_obj, epoch, use_wandb)
        val_loss, val_acc = validate(model, test_loader, criterion, device_obj, epoch)
        scheduler.step(val_loss)

        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

        if use_wandb and WANDB_AVAILABLE:
            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
            )

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
            logger.info(f"Saved best model (val_acc: {val_acc:.2f}%)")
            print(f"  ✓ Saved best model (val_acc: {val_acc:.2f}%)")

            if use_wandb and WANDB_AVAILABLE:
                artifact = wandb.Artifact(
                    name=f"model-{wandb.run.id}",
                    type="model",
                    description=f"Best model at epoch {epoch} with val_acc {val_acc:.2f}%",
                    metadata={
                        "epoch": epoch,
                        "val_acc": val_acc,
                        "val_loss": val_loss,
                        "train_acc": train_acc,
                        "train_loss": train_loss,
                    }
                )
                artifact.add_file(str(best_model_path))
                wandb.log_artifact(artifact)
                logger.info(f"Logged model artifact to W&B: {artifact.name}")

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
            logger.info(f"Saved checkpoint: {checkpoint_path.name}")
            print(f"  ✓ Saved checkpoint: {checkpoint_path.name}")

    logger.info("Training complete")
    logger.info(f"Best validation accuracy: {best_val_acc:.2f}%")
    logger.info(f"Best model saved to: {best_model_path}")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Best model saved to: {best_model_path}")

    if use_wandb and WANDB_AVAILABLE:
        wandb.finish()
        logger.info("Closed Weights & Biases connection")

if __name__ == "__main__":
    typer.run(train)
