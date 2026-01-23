import random
import time
from pathlib import Path

import requests
from rich.console import Console
from rich.progress import track

console = Console()


def send_batch_predictions(num_requests: int = 20) -> None:
    """
    Send multiple prediction requests to populate the database.

    Args:
        num_requests: Number of prediction requests to send
    """
    api_url = "http://localhost:8000"
    test_images = list(Path("raw/test").rglob("*.jpg"))[:100]

    if not test_images:
        console.print("[red]No test images found in raw/test/[/red]")
        return

    console.print(f"[cyan]Sending {num_requests} prediction requests...[/cyan]\n")

    successful = 0
    failed = 0

    for _ in track(range(num_requests), description="Sending requests"):
        image_path = random.choice(test_images)

        try:
            with open(image_path, "rb") as f:
                files = {"image": (image_path.name, f, "image/jpeg")}
                response = requests.post(f"{api_url}/predict", files=files, timeout=10)

            if response.status_code == 200:
                successful += 1
            else:
                failed += 1

            time.sleep(0.5)

        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            failed += 1

    console.print(f"\n[green]✓ Successful: {successful}[/green]")
    console.print(f"[red]✗ Failed: {failed}[/red]")
    console.print("\n[yellow]Now you can test the API endpoints manually.[/yellow]")
    console.print("[yellow]Run: uv run python scripts/test_api_client.py monitoring[/yellow]")


if __name__ == "__main__":
    send_batch_predictions(20)
