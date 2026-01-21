from pathlib import Path
from PIL import Image
import bentoml

from service import preprocess_pil

if __name__ == "__main__":
    image_path = Path("tests/test_photo/c4665dfa1f353f9a6b2571fadbb94e917dff2010-2100x1020.jpg")
    img = Image.open(image_path)

    img.show()  # optional: visually confirm what you’re testing

    arr = preprocess_pil(img)

    with bentoml.SyncHTTPClient("http://localhost:4040") as client:
        resp = client.predict(image=arr)
        print(resp[0])
