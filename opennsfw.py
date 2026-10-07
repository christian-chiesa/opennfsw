import os
import numpy as np
from PIL import Image
import onnxruntime as ort


def preprocess_image(image, target_size=(224, 224)):
    """
    Preprocesses an image for OpenNSFW inference.

    Args:
        image (PIL.Image.Image or str): PIL Image object or filepath.
        target_size (tuple): Target image resolution (width, height). Default (224, 224).

    Returns:
        np.ndarray: Preprocessed float32 numpy array with shape (1, 3, height, width).
    """
    if isinstance(image, (str, os.PathLike)):
        image = Image.open(image)

    image = image.convert("RGB")
    image = image.resize(target_size, Image.Resampling.BILINEAR)

    # Convert image to numpy array (H, W, C) in RGB order
    img_data = np.array(image, dtype=np.float32)

    # Convert RGB to BGR (standard Yahoo OpenNSFW format)
    img_data = img_data[:, :, ::-1]

    # Subtract VGG mean BGR values
    mean = np.array([104.0, 117.0, 123.0], dtype=np.float32)
    img_data -= mean

    # Transpose to (C, H, W)
    img_data = np.transpose(img_data, (2, 0, 1))

    # Add batch dimension: (1, C, H, W)
    img_data = np.expand_dims(img_data, axis=0)

    return img_data


class OpenNSFW:
    """
    OpenNSFW model wrapper using ONNX Runtime.
    """

    def __init__(self, model_path=None, session=None):
        """
        Initialize the OpenNSFW classifier.

        Args:
            model_path (str, optional): Path to the ONNX model file.
            session (ort.InferenceSession, optional): Pre-existing ONNX InferenceSession.
        """
        self.session = session
        self.model_path = model_path

        if self.session is None and self.model_path and os.path.exists(self.model_path):
            self.session = ort.InferenceSession(self.model_path)

    def predict_image(self, image):
        """
        Predict SFW and NSFW probabilities for a given image.

        Args:
            image (PIL.Image.Image or str): Image object or file path.

        Returns:
            dict: Dictionary with keys 'sfw' and 'nsfw' and their respective probability scores.
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


def predict(image, model_path):
    """
    Convenience function to classify an image using an ONNX model path.

    Args:
        image (PIL.Image.Image or str): Image or path to image.
        model_path (str): Path to OpenNSFW ONNX model.

    Returns:
        dict: SFW and NSFW probability scores.
    """
    classifier = OpenNSFW(model_path=model_path)
    return classifier.predict_image(image)
