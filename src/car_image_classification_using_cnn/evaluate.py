from pathlib import Path

import torch
import typer
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from car_image_classification_using_cnn.data import CarImageDataset, get_transforms
from car_image_classification_using_cnn.model import create_model


def evaluate(
    model_path: Path = typer.Argument(..., help="Path to the trained model checkpoint"),
    test_data_dir: Path = typer.Option(Path("raw/test"), help="Path to test data directory"),
    batch_size: int = typer.Option(32, help="Batch size for evaluation"),
    device: str = typer.Option("cuda" if torch.cuda.is_available() else "cpu", help="Device to use (cuda/cpu)"),
) -> None:
    """
    Evaluate a trained model on test data.

    Args:
        model_path: Path to the model checkpoint (.pth file)
        test_data_dir: Directory containing test images
        batch_size: Batch size for DataLoader
        device: Device to run evaluation on
    """
    # Load checkpoint
    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Get class names from checkpoint
    classes = checkpoint.get("classes", None)
    if classes is None:
        raise ValueError("Checkpoint does not contain class information")

    num_classes = len(classes)
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {classes}")

    # Create model and load weights
    model = create_model(num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    device_obj = torch.device(device)
    model = model.to(device_obj)
    model.eval()

    # Load test dataset
    print(f"\nLoading test data from {test_data_dir}")
    test_dataset = CarImageDataset(data_path=test_data_dir, transform=get_transforms(mode="test"))

    if set(test_dataset.classes) != set(classes):
        print("WARNING: Test dataset classes don't match training classes!")
        print(f"  Training classes: {classes}")
        print(f"  Test classes: {test_dataset.classes}")

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Evaluate
    print("\n" + "=" * 70)
    print("EVALUATION")
    print("=" * 70)

    correct = 0
    total = 0
    class_correct = {class_name: 0 for class_name in classes}
    class_total = {class_name: 0 for class_name in classes}

    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Evaluating")
        for images, labels in pbar:
            images, labels = images.to(device_obj), labels.to(device_obj)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Per-class accuracy
            for label, prediction in zip(labels, predicted):
                class_name = classes[label]
                class_total[class_name] += 1
                if label == prediction:
                    class_correct[class_name] += 1

    # Print results
    overall_accuracy = 100 * correct / total
    print(f"\nOverall Accuracy: {overall_accuracy:.2f}% ({correct}/{total})")

    print("\nPer-class Accuracy:")
    print("-" * 70)
    for class_name in classes:
        if class_total[class_name] > 0:
            class_acc = 100 * class_correct[class_name] / class_total[class_name]
            print(f"  {class_name:20s}: {class_acc:.2f}% ({class_correct[class_name]}/{class_total[class_name]})")
        else:
            print(f"  {class_name:20s}: No samples")

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)

    # Print checkpoint metadata if available
    if "epoch" in checkpoint:
        print(f"\nModel trained for {checkpoint['epoch']} epochs")
    if "val_acc" in checkpoint:
        print(f"Validation accuracy during training: {checkpoint['val_acc']:.2f}%")


if __name__ == "__main__":
    typer.run(evaluate)
