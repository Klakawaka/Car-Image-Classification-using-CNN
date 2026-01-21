from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

import main  # FastAPI app is in src/main.py


def _make_test_image_bytes(size=(64, 64)) -> bytes:
    img = Image.new("RGB", size, color=(120, 80, 200))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_root(api_env) -> None:
    # lifespan -> use TestClient as context manager
    with TestClient(main.app) as client:
        resp = client.get("/")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "active"
        assert body["documentation"] == "/docs"


def test_health(api_env) -> None:
    with TestClient(main.app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "healthy"}


def test_classes(api_env) -> None:
    with TestClient(main.app) as client:
        resp = client.get("/classes")
        assert resp.status_code == 200
        body = resp.json()
        assert body["num_classes"] == 6
        assert len(body["classes"]) == 6


def test_predict_rejects_non_image(api_env) -> None:
    with TestClient(main.app) as client:
        resp = client.post(
            "/predict",
            files={"image": ("not_an_image.txt", b"hello", "text/plain")},
        )
        assert resp.status_code == 400
        assert "must be an image" in resp.json()["detail"].lower()


def test_predict_success(api_env) -> None:
    img_bytes = _make_test_image_bytes()
    with TestClient(main.app) as client:
        resp = client.post(
            "/predict",
            files={"image": ("car.png", img_bytes, "image/png")},
        )
        assert resp.status_code == 200
        body = resp.json()

        assert body["predicted_class"] in {
            "Audi",
            "Hyundai Creta",
            "Rolls Royce",
            "Swift",
            "Tata Safari",
            "Toyota Innova",
        }
        assert 0.0 <= body["confidence"] <= 1.0

        probs = body["all_probabilities"]
        assert abs(sum(probs.values()) - 1.0) < 1e-4


def test_predict_batch_limit(api_env) -> None:
    img_bytes = _make_test_image_bytes()
    files = [("images", (f"car_{i}.png", img_bytes, "image/png")) for i in range(11)]

    with TestClient(main.app) as client:
        resp = client.post("/predict_batch", files=files)
        assert resp.status_code == 400
        assert "maximum 10" in resp.json()["detail"].lower()


def test_predict_batch_mixed_inputs(api_env) -> None:
    img_bytes = _make_test_image_bytes()
    files = [
        ("images", ("ok.png", img_bytes, "image/png")),
        ("images", ("bad.txt", b"nope", "text/plain")),
    ]

    with TestClient(main.app) as client:
        resp = client.post("/predict_batch", files=files)
        assert resp.status_code == 200
        body = resp.json()

        assert body["total"] == 2
        assert len(body["predictions"]) == 2

        ok, bad = body["predictions"]
        assert ok["filename"] == "ok.png"
        assert "predicted_class" in ok

        assert bad["filename"] == "bad.txt"
        assert "error" in bad
