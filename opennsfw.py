"""
OpenNSFW Python implementation using ONNX Runtime.
"""

import argparse
import os
import sys
import urllib.request
from typing import Dict, Union
import numpy as np
import onnxruntime as ort
from PIL import Image

MODEL_URL = "https://huggingface.co/jojonocode/mod_open_nsfw/resolve/main/open_nsfw.onnx"
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "open_nsfw.onnx")

# Yahoo OpenNSFW mean BGR values
VGG_MEAN = np.array([104.00698793, 116.66876762, 122.67891434], dtype=np.float32)


def download_model(model_path: str = DEFAULT_MODEL_PATH, url: str = MODEL_URL) -> str:
    """Download the ONNX model if it does not exist locally."""
    if not os.path.exists(model_path):
        os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp, open(model_path, "wb") as f:
            f.write(resp.read())
    return model_path


def preprocess_image(image: Union[str, Image.Image]) -> np.ndarray:
    """
    Preprocess image for Yahoo OpenNSFW ONNX model.
    1. Load image and convert to RGB.
    2. Resize to 256x256 using bilinear interpolation.
    3. Center crop to 224x224.
    4. Convert RGB to BGR.
    5. Subtract VGG mean values.
    6. Expand dimensions to shape (1, 224, 224, 3) with float32 type.
    """
    if isinstance(image, str):
        img = Image.open(image).convert("RGB")
    elif isinstance(image, Image.Image):
        img = image.convert("RGB")
    else:
        raise ValueError("Unsupported image type. Provide a file path or a PIL.Image.Image instance.")

    # Resize to 256x256
    img = img.resize((256, 256), Image.Resampling.BILINEAR)

    # Center crop 224x224
    left = (256 - 224) // 2
    top = (256 - 224) // 2
    right = left + 224
    bottom = top + 224
    img = img.crop((left, top, right, bottom))

    # Convert PIL Image to numpy array (H, W, C) float32 in range [0, 255]
    img_data = np.array(img, dtype=np.float32)

    # RGB to BGR
    img_data = img_data[:, :, ::-1]

    # Subtract VGG mean
    img_data -= VGG_MEAN

    # Add batch dimension -> (1, 224, 224, 3)
    img_data = np.expand_dims(img_data, axis=0)

    return img_data


class NSFWClassifier:
    """NSFW Classifier session wrapper using ONNX Runtime."""

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        if not os.path.exists(model_path):
            download_model(model_path)
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict(self, image: Union[str, Image.Image]) -> Dict[str, float]:
        """Classify an image and return NSFW/SFW probabilities."""
        preprocessed = preprocess_image(image)
        outputs = self.session.run([self.output_name], {self.input_name: preprocessed})
        probs = outputs[0][0]  # shape (2,) -> [sfw_prob, nsfw_prob]
        sfw_prob = float(probs[0])
        nsfw_prob = float(probs[1])
        return {
            "sfw": sfw_prob,
            "nsfw": nsfw_prob,
        }


def predict(image: Union[str, Image.Image], model_path: str = DEFAULT_MODEL_PATH) -> Dict[str, float]:
    """Convenience function to classify a single image."""
    classifier = NSFWClassifier(model_path=model_path)
    return classifier.predict(image)


def main():
    parser = argparse.ArgumentParser(description="Classify image using OpenNSFW ONNX model.")
    parser.add_argument("image_path", help="Path to image file")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Path to ONNX model file")
    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(f"Error: image file '{args.image_path}' not found.", file=sys.stderr)
        sys.exit(1)

    result = predict(args.image_path, model_path=args.model_path)
    print(f"SFW score:  {result['sfw']:.4f}")
    print(f"NSFW score: {result['nsfw']:.4f}")


if __name__ == "__main__":
    main()
