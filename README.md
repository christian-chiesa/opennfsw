# OpenNSFW Python

An OpenNSFW (Not Safe For Work) image classification implementation using **ONNX Runtime**, **Pillow**, and **NumPy**.

## Installation

Install the required dependencies via `pip`:

```bash
pip install -r requirements.txt
```

## Usage

### Using the Python API

```python
from opennsfw import OpenNSFW, predict

# Option 1: Using the OpenNSFW class
classifier = OpenNSFW(model_path="path/to/opennsfw.onnx")
result = classifier.predict_image("image.jpg")

print(f"SFW Probability: {result['sfw']:.4f}")
print(f"NSFW Probability: {result['nsfw']:.4f}")

# Option 2: Using the convenience predict function
result = predict("image.jpg", model_path="path/to/opennsfw.onnx")
```

## Running Tests

Run unit tests using Python's `unittest` framework:

```bash
python3 -m unittest test_opennsfw.py
```
