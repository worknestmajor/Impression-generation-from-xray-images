import io
import json
import unittest

import numpy as np
from PIL import Image

import inference_service as inference
from create_model import _format_impression


class NamedBytesIO(io.BytesIO):
    name = "test.png"


def png_upload(width=224,height=224):
    buffer = NamedBytesIO()
    Image.new("RGB",(width,height),color=(80,90,100)).save(buffer,format="PNG")
    buffer.seek(0)
    return buffer


class InferenceServiceTests(unittest.TestCase):
    def test_valid_image_returns_metadata_and_pixels(self):
        result = inference.inspect_image(png_upload())
        self.assertTrue(result["valid"])
        self.assertEqual((result["width"],result["height"]),(224,224))
        self.assertEqual(result["format"],"PNG")
        self.assertEqual(result["pixels"].shape,(224,224,3))

    def test_invalid_image_is_rejected(self):
        invalid = NamedBytesIO(b"not an image")
        result = inference.inspect_image(invalid)
        self.assertFalse(result["valid"])
        self.assertIn("Unable to read",result["error"])

    def test_low_resolution_adds_warning(self):
        result = inference.inspect_image(png_upload(64,64))
        self.assertTrue(result["valid"])
        self.assertTrue(any("Low resolution" in warning for warning in result["warnings"]))

    def test_normalization_is_float_and_bounded(self):
        result = inference.inspect_image(png_upload())
        pixels = inference.normalized_pixels(result)
        self.assertEqual(pixels.dtype,np.float32)
        self.assertGreaterEqual(float(pixels.min()),0.0)
        self.assertLessEqual(float(pixels.max()),1.0)

    def test_attention_overlay_preserves_image_shape(self):
        image = np.full((224,224,3),0.5,dtype=np.float32)
        attention = np.arange(18,dtype=np.float32)
        overlay = inference.attention_overlay(image,attention,0)
        self.assertEqual(overlay.shape,image.shape)
        self.assertEqual(overlay.dtype,np.uint8)

    def test_report_exports_are_serializable(self):
        report = {
            "id":"1","created_at":"2026-01-01T00:00:00+05:30",
            "impression":"No acute disease.","elapsed_seconds":1.25,"view_count":1,
            "primary":{"name":"front.png","format":"PNG","width":224,"height":224},"secondary":None,
            "primary_pixels":np.full((224,224,3),0.5,dtype=np.float32),
            "secondary_pixels":np.full((224,224,3),0.5,dtype=np.float32),
        }
        self.assertIn("No acute disease.",inference.report_text(report))
        exported = json.loads(inference.report_json(report))
        self.assertEqual(exported["views_supplied"],1)
        pdf = inference.report_pdf(report,"Edited impression.")
        self.assertTrue(pdf.startswith(b"%PDF-"))
        self.assertGreater(len(pdf),1000)

    def test_impression_formatting(self):
        actual = _format_impression("mild opacity . no acute infiltrate .")
        self.assertEqual(actual,"Mild opacity. No acute infiltrate.")


if __name__ == "__main__":
    unittest.main()
