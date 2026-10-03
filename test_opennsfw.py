"""
Unit tests for opennsfw module.
"""

import os
import tempfile
import unittest
import numpy as np
from PIL import Image

import opennsfw


class TestOpenNSFW(unittest.TestCase):

    def setUp(self):
        self.test_img = Image.new("RGB", (300, 300), color=(128, 64, 32))
        self.temp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        self.test_img.save(self.temp_file.name)
        self.temp_file.close()

    def tearDown(self):
        if os.path.exists(self.temp_file.name):
            os.remove(self.temp_file.name)

    def test_preprocess_image_pil(self):
        tensor = opennsfw.preprocess_image(self.test_img)
        self.assertIsInstance(tensor, np.ndarray)
        self.assertEqual(tensor.shape, (1, 224, 224, 3))
        self.assertEqual(tensor.dtype, np.float32)

    def test_preprocess_image_path(self):
        tensor = opennsfw.preprocess_image(self.temp_file.name)
        self.assertIsInstance(tensor, np.ndarray)
        self.assertEqual(tensor.shape, (1, 224, 224, 3))
        self.assertEqual(tensor.dtype, np.float32)

    def test_preprocess_invalid_input(self):
        with self.assertRaises(ValueError):
            opennsfw.preprocess_image(12345)

    def test_predict(self):
        res = opennsfw.predict(self.test_img)
        self.assertIsInstance(res, dict)
        self.assertIn("sfw", res)
        self.assertIn("nsfw", res)
        self.assertIsInstance(res["sfw"], float)
        self.assertIsInstance(res["nsfw"], float)
        self.assertAlmostEqual(res["sfw"] + res["nsfw"], 1.0, places=4)

    def test_classifier_class(self):
        classifier = opennsfw.NSFWClassifier()
        res = classifier.predict(self.temp_file.name)
        self.assertIsInstance(res, dict)
        self.assertIn("sfw", res)
        self.assertIn("nsfw", res)
        self.assertIsInstance(res["sfw"], float)
        self.assertIsInstance(res["nsfw"], float)


if __name__ == "__main__":
    unittest.main()
