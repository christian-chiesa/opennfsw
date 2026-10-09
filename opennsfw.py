import argparse
import os
import sys
import urllib.request
from typing import Dict, Union, Optional
import numpy as np
from PIL import Image
import onnxruntime as ort

MODEL_URL = "https://huggingface.co/jojonocode/mod_open_nsfw/resolve/main/open_nsfw.onnx"
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "open_nsfw.onnx")

# Yahoo OpenNSFW mean BGR values
VGG_MEAN = np.array([104.0, 117.0, 123.0], dtype=np.float32)


def download_model(model_path: str = DEFAULT_MODEL_PATH, url: str = MODEL_URL) -> str:
    """
    Download the OpenNSFW ONNX model if it does not exist locally.

    Args:
        model_path (str): Filepath where the model should be saved.
        url (str): Remote URL to download the model from.

    Returns:
        str: Filepath to downloaded model.
    """
    if not os.path.exists(model_path):
        os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp, open(model_path, "wb") as f:
            f.write(resp.read())
    return model_path


def preprocess_image(image: Union[str, os.PathLike, Image.Image], target_size=(224, 224)) -> np.ndarray:
    """
    Preprocesses an image for OpenNSFW model inference.

    Args:
        image (PIL.Image.Image, str, or os.PathLike): PIL Image instance or path to image file.
        target_size (tuple): Target image size (width, height). Default is (224, 224).

    Returns:
        np.ndarray: Preprocessed float32 numpy array with shape (1, 3, height, width).
    """
    if isinstance(image, (str, os.PathLike)):
        image = Image.open(image)
    elif not isinstance(image, Image.Image):
        raise ValueError("Unsupported image type. Provide a file path or PIL.Image.Image instance.")

    img = image.convert("RGB")
    img = img.resize(target_size, Image.Resampling.BILINEAR)

    # Convert PIL image to numpy array (H, W, C) in RGB order
    img_data = np.array(img, dtype=np.float32)

    # Convert RGB to BGR order
    img_data = img_data[:, :, ::-1]

    # Subtract VGG BGR mean
    img_data -= VGG_MEAN

    # Transpose HWC to CHW: (3, H, W)
    img_data = np.transpose(img_data, (2, 0, 1))

    # Add batch dimension: (1, 3, H, W)
    img_data = np.expand_dims(img_data, axis=0)

    return img_data


class OpenNSFW:
    """
    OpenNSFW classifier wrapper using ONNX Runtime.
    """

    def __init__(self, model_path: Optional[str] = None, session: Optional[ort.InferenceSession] = None):
        """
        Initialize the OpenNSFW classifier.

        Args:
            model_path (str, optional): Path to the ONNX model file.
            session (ort.InferenceSession, optional): Pre-existing ONNX InferenceSession.
        """
        self.session = session
        self.model_path = model_path

        if self.session is None and self.model_path:
            if not os.path.exists(self.model_path):
                download_model(self.model_path)
            if os.path.exists(self.model_path):
                self.session = ort.InferenceSession(self.model_path)

    def predict_image(self, image: Union[str, os.PathLike, Image.Image]) -> Dict[str, float]:
        """
        Predict SFW and NSFW probability scores for an image.

        Args:
            image (PIL.Image.Image, str, or os.PathLike): Image object or file path.

        Returns:
            dict: Dictionary containing 'sfw' and 'nsfw' probability scores.
        """
        if self.session is None:
            raise ValueError("InferenceSession is not initialized. Provide a valid model path or session.")

        input_tensor = preprocess_image(image)
        input_name = self.session.get_inputs()[0].name

        outputs = self.session.run(None, {input_name: input_tensor})
        probs = outputs[0][0]

        # Apply softmax if outputs are raw logits (values out of [0, 1] range or sum not close to 1)
        if np.any(probs < 0) or np.any(probs > 1) or not np.isclose(np.sum(probs), 1.0, atol=1e-2):
            exp_probs = np.exp(probs - np.max(probs))
            probs = exp_probs / np.sum(exp_probs)

        sfw_score = float(probs[0])
        nsfw_score = float(probs[1])

        return {
            "sfw": sfw_score,
            "nsfw": nsfw_score
        }


def predict(image: Union[str, os.PathLike, Image.Image], model_path: Optional[str] = None) -> Dict[str, float]:
    """
    Convenience function to classify an image using an ONNX model.

    Args:
        image (PIL.Image.Image, str, or os.PathLike): Image or path to image.
        model_path (str, optional): Path to OpenNSFW ONNX model. Defaults to DEFAULT_MODEL_PATH.

    Returns:
        dict: SFW and NSFW probability scores.
    """
    if model_path is None:
        model_path = DEFAULT_MODEL_PATH
    classifier = OpenNSFW(model_path=model_path)
    return classifier.predict_image(image)


def main():
    parser = argparse.ArgumentParser(description="Classify image using OpenNSFW ONNX model.")
    parser.add_argument("image_path", help="Path to image file")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Path to ONNX model file")
    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(f"Error: Image file '{args.image_path}' not found.", file=sys.stderr)
        sys.exit(1)

    result = predict(args.image_path, model_path=args.model_path)
    print(f"SFW score:  {result['sfw']:.4f}")
    print(f"NSFW score: {result['nsfw']:.4f}")


if __name__ == "__main__":
    main()
