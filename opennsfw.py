"""
OpenNSFW Model & Classification Utility

Provides standard image preprocessing and NSFW classification interface compatible
with Yahoo OpenNSFW model architecture (ResNet-50).
"""

import argparse
import json
import os
import sys
from typing import Dict, Union, Tuple, Optional
import numpy as np
from PIL import Image

# Standard OpenNSFW mean values for BGR image preprocessing
VGG_MEAN = np.array([104.00698793, 116.66876762, 122.67891434], dtype=np.float32)


def preprocess_image(
    image_path: Union[str, Image.Image],
    target_size: Tuple[int, int] = (224, 224)
) -> np.ndarray:
    """
    Preprocess an image for the Yahoo OpenNSFW model.
    1. Open image and convert to RGB.
    2. Resize image to target_size (224, 224).
    3. Convert to float32 numpy array.
    4. Convert RGB to BGR.
    5. Subtract Mean (VGG_MEAN: [104.0, 116.7, 122.7]).
    6. Expand dimensions to (1, H, W, C).

    Args:
        image_path: Path to input image file or PIL Image object.
        target_size: Tuple of (height, width) expected by the network.

    Returns:
        np.ndarray: Preprocessed image tensor array.
    """
    if isinstance(image_path, str):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at path: {image_path}")
        img = Image.open(image_path)
    else:
        img = image_path

    img = img.convert("RGB")
    img = img.resize(target_size, Image.Resampling.BILINEAR)

    # Convert to numpy array (H, W, C) float32
    img_data = np.array(img, dtype=np.float32)

    # RGB to BGR
    img_bgr = img_data[:, :, ::-1]

    # Subtract VGG mean
    img_normalized = img_bgr - VGG_MEAN

    # Add batch dimension: (1, H, W, C)
    batch_data = np.expand_dims(img_normalized, axis=0)

    return batch_data


class OpenNSFWModel:
    """
    OpenNSFW Model Classifier Interface.
    Loads ONNX or Caffe OpenNSFW weights and evaluates images.
    """

    def __init__(self, model_weights_path: Optional[str] = None):
        """
        Initialize OpenNSFW Model.

        Args:
            model_weights_path: Path to .onnx model file.
        """
        self.model_weights_path = model_weights_path
        self._session = None

        if model_weights_path:
            if not os.path.exists(model_weights_path):
                raise FileNotFoundError(f"Weights file not found at: {model_weights_path}")
            self._load_model(model_weights_path)

    def _load_model(self, weights_path: str):
        """Helper to load ONNX session."""
        if weights_path.endswith(".onnx"):
            try:
                import onnxruntime as ort
                self._session = ort.InferenceSession(weights_path)
            except ImportError:
                raise RuntimeError(
                    "onnxruntime package is required to perform inference with ONNX model weights. "
                    "Please install it using 'pip install onnxruntime'."
                )
        else:
            raise ValueError(f"Unsupported model weight format: {weights_path}. Only .onnx model files are supported.")

    def predict(self, image_input: Union[str, Image.Image, np.ndarray]) -> Dict[str, float]:
        """
        Predict NSFW score for an image.

        Args:
            image_input: File path, PIL Image, or preprocessed numpy array.

        Returns:
            Dict[str, float]: Dictionary containing 'sfw' and 'nsfw' probability scores.
        """
        if isinstance(image_input, np.ndarray):
            tensor = image_input
        else:
            tensor = preprocess_image(image_input)

        if self._session is None:
            raise RuntimeError(
                "No OpenNSFW model weights loaded. Please provide a valid model path "
                "(e.g., OpenNSFWModel('path/to/open_nsfw.onnx')) or pass model_weights to classify_image()."
            )

        input_name = self._session.get_inputs()[0].name

        # Handle NCHW vs NHWC tensor shapes
        if len(self._session.get_inputs()[0].shape) == 4 and self._session.get_inputs()[0].shape[1] == 3:
            tensor_input = np.transpose(tensor, (0, 3, 1, 2))
        else:
            tensor_input = tensor

        outputs = self._session.run(None, {input_name: tensor_input})
        probs = outputs[0][0]

        # Softmax normalization if outputs are raw logits
        if not np.isclose(np.sum(probs), 1.0, atol=1e-2):
            e_x = np.exp(probs - np.max(probs))
            probs = e_x / e_x.sum()

        sfw_score = float(probs[0])
        nsfw_score = float(probs[1])

        return {
            "sfw": round(sfw_score, 4),
            "nsfw": round(nsfw_score, 4)
        }


def classify_image(image_path: str, model_weights: str) -> Dict[str, float]:
    """
    Convenience function to classify an image using specified model weights.

    Args:
        image_path: Path to image file.
        model_weights: Path to OpenNSFW .onnx weights file.

    Returns:
        Dict[str, float]: Dictionary with sfw and nsfw scores.
    """
    model = OpenNSFWModel(model_weights_path=model_weights)
    return model.predict(image_path)


def main():
    parser = argparse.ArgumentParser(description="OpenNSFW Image Classifier")
    parser.add_argument("image_path", help="Path to the image file to classify")
    parser.add_argument("--weights", required=True, help="Path to OpenNSFW model weights (.onnx)")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")

    args = parser.parse_args()

    try:
        results = classify_image(args.image_path, model_weights=args.weights)
        if args.json:
            print(json.dumps(results, indent=2))
        else:
            print(f"File: {args.image_path}")
            print(f"SFW Score:  {results['sfw']:.4f}")
            print(f"NSFW Score: {results['nsfw']:.4f}")
    except Exception as e:
        print(f"Error classifying image: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
