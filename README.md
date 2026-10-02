# opennsfw

A lightweight Python library and CLI tool for OpenNSFW image classification using ONNX Runtime.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Simple API

```python
import opennsfw

# Classify an image file path or PIL Image object
result = opennsfw.predict("image.jpg")
print(result)
# Output: {'sfw': 0.985, 'nsfw': 0.015}
```

### NSFWClassifier Object

```python
from opennsfw import NSFWClassifier

classifier = NSFWClassifier()
result = classifier.predict("image.jpg")
print(f"NSFW probability: {result['nsfw']:.4f}")
```

### CLI

```bash
python3 opennsfw.py image.jpg
```

## Features

- Uses ONNX Runtime for fast CPU/GPU inference.
- Auto-downloads the ONNX model (`open_nsfw.onnx`) if not present locally.
- Standard Yahoo OpenNSFW preprocessing (resize to 256x256, center crop to 224x224, RGB->BGR, VGG mean subtraction).
