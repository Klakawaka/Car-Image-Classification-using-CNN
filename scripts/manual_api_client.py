from pathlib import Path

import requests
import typer
from rich.console import Console

console = Console()


def test_predict_endpoint(
    image_path: Path = typer.Argument(..., help="Path to test image"),
    api_url: str = typer.Option("http://localhost:8000", help="API base URL"),
) -> None:
    """
    Test the /predict endpoint with a local image.

    Args:
        image_path: Path to the test image file
        api_url: Base URL of the API server
    """
    if not image_path.exists():
        console.print(f"[red]Error: Image not found at {image_path}[/red]")
        raise typer.Exit(1)

    console.print(f"[cyan]Testing API at: {api_url}[/cyan]")
    console.print(f"[cyan]Using image: {image_path}[/cyan]\n")

    try:
        with open(image_path, "rb") as f:
            files = {"image": (image_path.name, f, "image/jpeg")}
            response = requests.post(f"{api_url}/predict", files=files, timeout=30)

        if response.status_code == 200:
            result = response.json()
            console.print("[green]✓ Prediction successful![/green]\n")
            console.print(f"[bold]Predicted Class:[/bold] {result['predicted_class']}")
            console.print(f"[bold]Confidence:[/bold] {result['confidence']:.4f}\n")
            console.print("[bold]All Probabilities:[/bold]")
            for class_name, prob in result["all_probabilities"].items():
                console.print(f"  {class_name}: {prob:.4f}")
        else:
            console.print(f"[red]✗ Request failed with status {response.status_code}[/red]")
            console.print(f"[red]Error: {response.text}[/red]")

    except requests.exceptions.ConnectionError:
        console.print("[red]✗ Connection failed. Is the API server running?[/red]")
        console.print("[yellow]Start the server with: uv run uvicorn src.main:app --reload[/yellow]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]✗ Error: {str(e)}[/red]")
        raise typer.Exit(1)


def test_monitoring_endpoint(
    api_url: str = typer.Option("http://localhost:8000", help="API base URL"),
    n_samples: int = typer.Option(50, help="Number of samples for drift report"),
) -> None:
    """
    Test the /monitoring endpoint.

    Args:
        api_url: Base URL of the API server
        n_samples: Number of samples to use in drift report
    """
    console.print(f"[cyan]Fetching monitoring report from: {api_url}/monitoring[/cyan]\n")

    try:
        response = requests.get(f"{api_url}/monitoring?n={n_samples}", timeout=60)

        if response.status_code == 200:
            console.print("[green]✓ Monitoring report generated successfully![/green]")

            # Save the HTML report
            output_path = Path("monitoring_report_test.html")
            with open(output_path, "w") as f:
                f.write(response.text)

            console.print(f"[green]Report saved to: {output_path}[/green]")
            console.print("[yellow]Open it in your browser to view the drift analysis[/yellow]")
        elif response.status_code == 404:
            console.print("[yellow]⚠ No predictions logged yet. Send some predictions first.[/yellow]")
        else:
            console.print(f"[red]✗ Request failed with status {response.status_code}[/red]")
            console.print(f"[red]Error: {response.text}[/red]")

    except requests.exceptions.ConnectionError:
        console.print("[red]✗ Connection failed. Is the API server running?[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]✗ Error: {str(e)}[/red]")
        raise typer.Exit(1)


def main() -> None:
    """Main CLI application."""
    app = typer.Typer(help="Test the Car Classification API endpoints")
    app.command(name="predict")(test_predict_endpoint)
    app.command(name="monitoring")(test_monitoring_endpoint)
    app()


if __name__ == "__main__":
    main()
