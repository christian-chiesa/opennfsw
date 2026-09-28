# opennsfw

OpenNSFW implementation and image classification utility.

This repository provides Python modules and CLI tools for scoring images using the Yahoo OpenNSFW standard image preprocessing and classification pipeline.

## Requirements

- Python 3.8+
- `Pillow`
- `numpy`
- `onnxruntime` (for inference with ONNX OpenNSFW models)

Install required dependencies:

```bash
pip install pillow numpy onnxruntime
```

## Usage

### Command Line Interface (CLI)

Classify an image file using OpenNSFW ONNX model weights:

```bash
python3 opennsfw.py path/to/image.jpg --weights path/to/open_nsfw.onnx
```

Output as JSON:

```bash
python3 opennsfw.py path/to/image.jpg --weights path/to/open_nsfw.onnx --json
```

### Python Library API

```python
from opennsfw import classify_image, OpenNSFWModel, preprocess_image

# Quick classification using ONNX model weights
results = classify_image("sample.jpg", model_weights="path/to/open_nsfw.onnx")
print(f"SFW Score: {results['sfw']}, NSFW Score: {results['nsfw']}")

# Advanced usage with preprocessed numpy batch input
model = OpenNSFWModel(model_weights_path="path/to/open_nsfw.onnx")
image_tensor = preprocess_image("sample.jpg")
scores = model.predict(image_tensor)
print(scores)
```

## Model Architecture & Preprocessing

OpenNSFW relies on a ResNet-50 residual network architecture. Preprocessing steps are:
1. Resize image to $224 \times 224$ pixels using bilinear interpolation.
2. Convert image color channels from RGB to BGR.
3. Subtract VGG mean values `[104.00698793, 116.66876762, 122.67891434]`.
4. Output probability scores for SFW and NSFW classes.

## License

GNU General Public License v3.0 (GPL-3.0)
