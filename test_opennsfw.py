import unittest
from unittest.mock import MagicMock
import numpy as np
from PIL import Image
import opennsfw


class TestOpenNSFW(unittest.TestCase):

    def setUp(self):
        # Create a simple test RGB image
        self.test_img = Image.new("RGB", (300, 300), color=(255, 0, 0))

    def test_preprocess_image_shape_and_dtype(self):
        tensor = opennsfw.preprocess_image(self.test_img)
        self.assertIsInstance(tensor, np.ndarray)
        self.assertEqual(tensor.dtype, np.float32)
        # Expected shape: (1, 3, 224, 224)
        self.assertEqual(tensor.shape, (1, 3, 224, 224))

    def test_preprocess_image_from_file(self):
        # Save image to temporary path and preprocess
        test_path = "temp_test_image.jpg"
        self.test_img.save(test_path)
        try:
            tensor = opennsfw.preprocess_image(test_path)
            self.assertEqual(tensor.shape, (1, 3, 224, 224))
        finally:
            import os
            if os.path.exists(test_path):
                os.remove(test_path)

    def test_open_nsfw_predict_image_mock_session_logits(self):
        # Mock session returning raw logits [1.0, 2.0]
        mock_session = MagicMock()
        mock_input = MagicMock()
        mock_input.name = "input"
        mock_session.get_inputs.return_value = [mock_input]

        # Simulated raw logits: SFW logit=1.0, NSFW logit=2.0
        mock_session.run.return_value = [np.array([[1.0, 2.0]], dtype=np.float32)]

        classifier = opennsfw.OpenNSFW(session=mock_session)
        res = classifier.predict_image(self.test_img)

        self.assertIn("sfw", res)
        self.assertIn("nsfw", res)
        self.assertAlmostEqual(res["sfw"] + res["nsfw"], 1.0, places=5)
        self.assertGreater(res["nsfw"], res["sfw"])

    def test_open_nsfw_predict_image_mock_session_probabilities(self):
        mock_session = MagicMock()
        mock_input = MagicMock()
        mock_input.name = "input"
        mock_session.get_inputs.return_value = [mock_input]

        # Simulated softmax probabilities
        mock_session.run.return_value = [np.array([[0.85, 0.15]], dtype=np.float32)]

        classifier = opennsfw.OpenNSFW(session=mock_session)
        res = classifier.predict_image(self.test_img)

        self.assertAlmostEqual(res["sfw"], 0.85, places=5)
        self.assertAlmostEqual(res["nsfw"], 0.15, places=5)

    def test_predict_image_uninitialized_session_raises(self):
        classifier = opennsfw.OpenNSFW()
        with self.assertRaises(ValueError):
            classifier.predict_image(self.test_img)


if __name__ == "__main__":
    unittest.main()
