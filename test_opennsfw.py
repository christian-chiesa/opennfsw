import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from PIL import Image
import opennsfw


class TestOpenNSFW(unittest.TestCase):

    def setUp(self):
        # Create a temporary dummy image (200x200 RGB)
        self.dummy_img = Image.new("RGB", (200, 200), color=(120, 130, 140))

    def test_preprocess_image_shape(self):
        tensor = opennsfw.preprocess_image(self.dummy_img)
        self.assertIsInstance(tensor, np.ndarray)
        # Expected shape: (1, 224, 224, 3)
        self.assertEqual(tensor.shape, (1, 224, 224, 3))

    def test_preprocess_image_mean_subtraction(self):
        tensor = opennsfw.preprocess_image(self.dummy_img)
        # First pixel RGB was (120, 130, 140) -> BGR is (140, 130, 120)
        # VGG_MEAN is [104.00698793, 116.66876762, 122.67891434]
        expected_b = 140.0 - 104.00698793
        expected_g = 130.0 - 116.66876762
        expected_r = 120.0 - 122.67891434

        np.testing.assert_allclose(
            tensor[0, 0, 0, :],
            [expected_b, expected_g, expected_r],
            rtol=1e-4
        )

    def test_missing_file_raises_error(self):
        with self.assertRaises(FileNotFoundError):
            opennsfw.preprocess_image("non_existent_file.jpg")

    def test_unloaded_model_predict_raises_runtime_error(self):
        model = opennsfw.OpenNSFWModel()
        with self.assertRaises(RuntimeError):
            model.predict(self.dummy_img)

    def test_missing_weights_file_raises_error(self):
        with self.assertRaises(FileNotFoundError):
            opennsfw.OpenNSFWModel("non_existent_model.onnx")

    def test_unsupported_weights_format_raises_error(self):
        with patch("os.path.exists", return_value=True):
            with self.assertRaises(ValueError):
                opennsfw.OpenNSFWModel("model.invalid")

    def test_mock_onnx_inference(self):
        mock_input = MagicMock()
        mock_input.name = "input:0"
        mock_input.shape = [1, 224, 224, 3]

        mock_session = MagicMock()
        mock_session.get_inputs.return_value = [mock_input]
        mock_session.run.return_value = [np.array([[0.8, 0.2]])]

        model = opennsfw.OpenNSFWModel()
        model._session = mock_session

        results = model.predict(self.dummy_img)
        self.assertEqual(results["sfw"], 0.8)
        self.assertEqual(results["nsfw"], 0.2)


if __name__ == "__main__":
    unittest.main()
