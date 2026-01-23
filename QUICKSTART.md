# Quick Start Guide

## 1 Activate environment

```bash
cd Car-Image-Classification-using-CNN
source .venv/bin/activate
```

(If you use uv, activation is optional but still fine.)

## 2 Pull data from DVC (dataset & models if tracked)

```bash
uv run dvc pull
```

Verify data exists:

```bash
ls raw/train
ls raw/test
```

## 3 Train the model (improved training)

```bash
uv run python -m car_image_classification_using_cnn.train \
  --model-type resnet \
  --pretrained \
  --num-epochs 20 \
  --batch-size 32 \
  --learning-rate 0.0003 \
  --weight-decay 0.0001 \
  --device auto
```

📌 Output:
- Saves `models/best_model.pth`
- Logs training & validation accuracy

## 4 Export trained model to ONNX (for deployment)

```bash
uv run python scripts/export_onnx.py
```

Verify:

```bash
ls -lh models/best_model.onnx
```

## 5 Run BentoML service locally (without Docker)

```bash
uv run bentoml serve service:CarClassifierService --port 4040 --host 0.0.0.0
```

Test:

```bash
curl http://localhost:4040/schema.json
```

## 6 Test BentoML API with client script

```bash
uv run python client_bento.py
```

Expected output:

```json
{'predicted_class': 'Audi', 'confidence': 0.99, ...}
```

## 7 Build & run EVERYTHING with Docker (backend + frontend)

```bash
docker compose build --no-cache
docker compose up
```

Services:
- Backend (BentoML): http://localhost:4040
- Frontend (Streamlit): http://localhost:8080

## 8 Use the frontend

Open browser: http://localhost:8080

Steps:
1. Upload image (PNG / JPG)
2. Click Predict
3. See prediction + confidence

## 9 Re-train & redeploy (full cycle)

Whenever you improve training:

```bash
uv run python -m car_image_classification_using_cnn.train --pretrained
uv run python scripts/export_onnx.py
docker compose build --no-cache backend
docker compose up
```

## 10 Optional debugging commands

Check class balance:

```bash
find raw/train -type f \( -iname "*.jpg" -o -iname "*.png" \) \
| sed 's|raw/train/||' | cut -d/ -f1 | sort | uniq -c
```

Inspect labels visually:

```bash
python - <<'PY'
from pathlib import Path
import random
from PIL import Image

root = Path("raw/train")
for cls in root.iterdir():
    imgs = list(cls.glob("*.jpg")) + list(cls.glob("*.png"))
    if imgs:
        Image.open(random.choice(imgs)).show()
PY
```
