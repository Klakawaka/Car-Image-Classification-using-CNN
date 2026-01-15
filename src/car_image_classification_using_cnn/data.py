import shutil
from pathlib import Path
from typing import Tuple

import torch
import typer
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class CarImageDataset(Dataset):
    """
    Custom dataset for car image classification.

    This dataset loads images from a directory structure where each subdirectory
    represents a different car class/brand.
    """

    def __init__(
        self,
        data_path: Path,
        transform: transforms.Compose | None = None,
        image_size: Tuple[int, int] = (224, 224),
    ) -> None:
        """
        Initialize the car image dataset.

        Args:
            data_path: Path to the directory containing subdirectories of car images.
            transform: Optional torchvision transforms to apply to images.
            image_size: Target size for images (height, width).
        """
        self.data_path = Path(data_path)
        self.image_size = image_size

        if transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(self.image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        else:
            self.transform = transform

        # Build dataset index
        self.samples = []
        self.class_to_idx = {}
        self.classes = []

        if self.data_path.exists():
            self._build_dataset()
        else:
            raise FileNotFoundError(f"Data path does not exist: {self.data_path}")

    def _build_dataset(self) -> None:
        """Scan the directory structure and build the dataset index."""
        # Get all class directories (car brands)
        class_dirs = sorted([d for d in self.data_path.iterdir() if d.is_dir()])

        # Create class to index mapping
        self.classes = [d.name for d in class_dirs]
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.classes)}

        # Collect all image paths and their labels
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        corrupted_count = 0
        for class_dir in class_dirs:
            class_idx = self.class_to_idx[class_dir.name]
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in valid_extensions:
                    # Try to verify the image can be opened
                    try:
                        with Image.open(img_path) as img:
                            img.verify()
                        self.samples.append((img_path, class_idx))
                    except Exception:
                        corrupted_count += 1

        if corrupted_count > 0:
            print(f"Warning: Skipped {corrupted_count} corrupted images during dataset initialization.")

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """
        Return a given sample from the dataset.

        Args:
            index: Index of the sample to return.

        Returns:
            Tuple of (image_tensor, label) where image_tensor is a transformed image
            and label is the class index.
        """
        img_path, label = self.samples[index]

        # Load image
        image = Image.open(img_path).convert("RGB")

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        return image, label

    def preprocess(self, output_folder: Path) -> None:
        """
        Preprocess the raw data and save it to the output folder.

        This method copies the data from the raw directory to the output folder,
        maintaining the directory structure. It can be extended to perform
        additional preprocessing like data augmentation, filtering, etc.

        Args:
            output_folder: Path where the preprocessed data should be saved.
        """
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

        print(f"Preprocessing data from {self.data_path}")
        print(f"Saving to {output_folder}")

        # Copy directory structure
        if self.data_path.exists():
            # Iterate through each class directory
            for class_dir in self.data_path.iterdir():
                if class_dir.is_dir():
                    dest_class_dir = output_folder / class_dir.name
                    dest_class_dir.mkdir(parents=True, exist_ok=True)

                    # Copy all valid images
                    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
                    copied_count = 0
                    for img_path in class_dir.iterdir():
                        if img_path.suffix.lower() in valid_extensions:
                            dest_path = dest_class_dir / img_path.name
                            if not dest_path.exists():
                                shutil.copy2(img_path, dest_path)
                                copied_count += 1

                    print(f"  Copied {copied_count} images for class '{class_dir.name}'")

        print("Preprocessing complete!")


def get_transforms(mode: str = "train", image_size: Tuple[int, int] = (224, 224)) -> transforms.Compose:
    """
    Get appropriate transforms for training or evaluation.

    Args:
        mode: Either 'train' or 'eval' to determine augmentation strategy.
        image_size: Target size for images (height, width).

    Returns:
        Composed transforms for the specified mode.
    """
    if mode == "train":
        return transforms.Compose(
            [
                transforms.Resize((int(image_size[0] * 1.1), int(image_size[1] * 1.1))),
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )


def preprocess(data_path: Path, output_folder: Path) -> None:
    """
    Preprocess raw data and save to output folder.

    Args:
        data_path: Path to the raw data directory.
        output_folder: Path where processed data should be saved.
    """
    print("Preprocessing data...")
    dataset = CarImageDataset(data_path)

    print(f"Found {len(dataset)} images across {len(dataset.classes)} classes")
    print(f"Classes: {dataset.classes}")

    dataset.preprocess(output_folder)


if __name__ == "__main__":
    typer.run(preprocess)
