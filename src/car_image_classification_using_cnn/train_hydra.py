from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn, optim
from torch.utils.data import DataLoader

from car_image_classification_using_cnn.data import CarImageDataset, get_transforms
from car_image_classification_using_cnn.model import create_model
from car_image_classification_using_cnn.train import train_one_epoch, validate


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if device == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print("Loaded config:\n", OmegaConf.to_yaml(cfg))

    device = resolve_device(cfg.training.device)
    print(f"Using device: {device}")

    # Output dir (Hydra changes working directory by default; make paths explicit)
    output_dir = Path(cfg.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Datasets
    train_dataset = CarImageDataset(
        data_path=Path(cfg.data.train_data_path),
        transform=get_transforms("train", tuple(cfg.data.image_size)),
        image_size=tuple(cfg.data.image_size),
    )
    test_dataset = CarImageDataset(
        data_path=Path(cfg.data.test_data_path),
        transform=get_transforms("eval", tuple(cfg.data.image_size)),
        image_size=tuple(cfg.data.image_size),
    )

    if train_dataset.classes != test_dataset.classes:
        raise ValueError(f"Train/Test class mismatch!\nTrain: {train_dataset.classes}\nTest: {test_dataset.classes}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        pin_memory=device.type in ("cuda", "mps"),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=device.type in ("cuda", "mps"),
    )

    # Model
    model = create_model(
        model_type=cfg.model.model_type,
        num_classes=len(train_dataset.classes),
        pretrained=cfg.model.pretrained,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    best_val_acc = 0.0
    best_model_path = output_dir / "best_model.pth"

    for epoch in range(1, cfg.training.num_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_acc = validate(model, test_loader, criterion, device, epoch)
        scheduler.step(val_loss)

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
                    "cfg": OmegaConf.to_container(cfg, resolve=True),
                },
                best_model_path,
            )

    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Best model saved to: {best_model_path}")


if __name__ == "__main__":
    main()
