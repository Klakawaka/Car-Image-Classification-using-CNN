from pathlib import Path
from PIL import Image, ImageOps


def convert_all_to_grayscale(input_root: Path, output_root: Path) -> None:
    input_root = Path(input_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # Iterate over all subfolders (brands)
    for class_dir in input_root.iterdir():
        if class_dir.is_dir():
            output_class_dir = output_root / class_dir.name
            output_class_dir.mkdir(parents=True, exist_ok=True)

            jpg_files = list(class_dir.glob("*.jpg"))
            if not jpg_files:
                print(f"No JPG images found in {class_dir}")
                continue

            for img_path in jpg_files:
                with Image.open(img_path) as img:
                    gray_img = img.convert("L")  # Grayscale
                    normalized_img = ImageOps.equalize(gray_img)  # Normalize lighting
                    normalized_img.save(output_class_dir / img_path.name)

            print(f"Converted and normalized {len(jpg_files)} images for class '{class_dir.name}'.")

    print("All images processed!")


if __name__ == "__main__":
    convert_all_to_grayscale(
        input_root=Path("/Users/sabirinomar/Car-Image-Classification-using-CNN/raw/test"),
        output_root=Path("/Users/sabirinomar/Car-Image-Classification-using-CNN/raw/test"),
    )
