from __future__ import annotations

import io
import os
import unittest
from unittest.mock import patch

from PIL import Image

from lmms_eval.models.model_utils.media_encoder import encode_image_to_bytes


class TestPngCompressLevel(unittest.TestCase):
    def test_png_compress_level_is_lossless(self):
        image = Image.frombytes("RGB", (64, 64), os.urandom(64 * 64 * 3))
        with patch.dict(os.environ, {"LMMS_IMAGE_PNG_COMPRESS_LEVEL": "1"}):
            fast = encode_image_to_bytes(image, image_format="PNG")
        default = encode_image_to_bytes(image, image_format="PNG")

        self.assertEqual(Image.open(io.BytesIO(fast)).tobytes(), image.tobytes())
        self.assertEqual(Image.open(io.BytesIO(default)).tobytes(), image.tobytes())
