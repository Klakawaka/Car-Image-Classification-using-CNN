from __future__ import annotations

import os
from io import BytesIO
from typing import Any

import pandas as pd
import requests
import streamlit as st
from PIL import Image


def get_backend_url() -> str:
    """
    Backend base URL.

    Locally: export BACKEND=http://localhost:4040
    Deployed: set BACKEND to your Cloud Run URL for the BentoML service.
    """
    backend = os.environ.get("BACKEND", "").strip()
    if not backend:
        # sensible local default
        backend = "http://localhost:4040"
    return backend.rstrip("/")


def preprocess_for_bento(img: Image.Image) -> list:
    """
    Must match your Bento service input: np.ndarray shaped (1,3,224,224).
    Streamlit/requests can't send numpy directly, so we send a nested list.
    """
    img = img.convert("RGB").resize((224, 224))

    # Convert to CHW float32 normalized like ImageNet (same as your service.py preprocess)
    import numpy as np

    x = np.asarray(img).astype("float32") / 255.0
    x = np.transpose(x, (2, 0, 1))  # HWC -> CHW
    mean = np.array([0.485, 0.456, 0.406], dtype="float32")[:, None, None]
    std = np.array([0.229, 0.224, 0.225], dtype="float32")[:, None, None]
    x = (x - mean) / std
    x = np.expand_dims(x, axis=0)  # (1,3,224,224)

    return x.tolist()  # JSON-serializable


def call_backend(image_array_as_list: list, backend: str) -> dict[str, Any]:
    """
    Calls BentoML endpoint.
    Your Bento service exposes POST /predict and returns a list[dict].
    """
    url = f"{backend}/predict"
    r = requests.post(url, json={"image": image_array_as_list}, timeout=30)
    r.raise_for_status()
    data = r.json()

    # because service returns list[dict], take first element
    if isinstance(data, list) and len(data) > 0:
        return data[0]
    raise ValueError(f"Unexpected response format: {type(data)}")


def main() -> None:
    st.set_page_config(page_title="Car Classifier", page_icon="🚗", layout="centered")
    st.title("🚗 Car Image Classification")
    st.write("Upload a car image and get the predicted class + probabilities.")

    backend = get_backend_url()
    st.caption(f"Backend: {backend}")

    uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    if uploaded is None:
        st.info("Upload an image to begin.")
        return

    # show image
    raw = uploaded.read()
    img = Image.open(BytesIO(raw))
    st.image(img, caption="Uploaded image", use_container_width=True)

    if st.button("Predict", type="primary"):
        with st.spinner("Running inference..."):
            try:
                image_payload = preprocess_for_bento(img)
                result = call_backend(image_payload, backend)

                st.success("Done!")
                st.write("**Prediction:**", result["predicted_class"])
                st.write("**Confidence:**", float(result["confidence"]))

                probs = result.get("all_probabilities", {})
                if isinstance(probs, dict) and probs:
                    df = pd.DataFrame([{"class": k, "probability": float(v)} for k, v in probs.items()]).sort_values(
                        "probability", ascending=False
                    )
                    st.bar_chart(df.set_index("class")["probability"])
                else:
                    st.warning("No probabilities returned from backend.")
            except requests.HTTPError as e:
                st.error(f"Backend error: {e.response.status_code} {e.response.text}")
            except Exception as e:
                st.error(f"Failed: {e}")


if __name__ == "__main__":
    main()
