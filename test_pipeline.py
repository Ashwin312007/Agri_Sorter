#!/usr/bin/env python3
"""Unit tests for Apple Damage Detector pipeline logic.

Tests:
1. letterbox: verifies dimensions, aspect ratio scaling, padding
2. preprocess: verifies shape [1, 3, 416, 416], normalization [0, 1], dtype float32
3. postprocess: verifies YOLOv5 synthetic output decoding, coordinate transformation, NMS
4. is_defect: verifies fresh vs reject decision rule
5. load_names: verifies metadata parsing, CLI parsing, and mismatch handling
6. dataset_prep remapping: verifies correct class consolidation and dropped labels
"""
import ast
import unittest
from unittest.mock import MagicMock

import numpy as np

# Import functions from detect.py
from detect import (
    APPLE_CLASSES,
    GUAVA_CLASSES,
    MANGO_CLASSES,
    is_defect,
    letterbox,
    load_names,
    postprocess,
    preprocess,
)
# Import functions from dataset_prep.py
from dataset_prep import UNIFIED_CLASSES, remap_class


class TestLetterbox(unittest.TestCase):
    def test_letterbox_dimensions_and_padding(self):
        # Create a synthetic 480x640 BGR image
        orig_h, orig_w = 480, 640
        img = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
        target_size = 416

        out, r, (left, top) = letterbox(img, target_size)

        # Output shape must be exactly (416, 416, 3)
        self.assertEqual(out.shape, (target_size, target_size, 3))

        # Scaling ratio should be min(416/480, 416/640) = 416/640 = 0.65
        expected_r = target_size / orig_w
        self.assertAlmostEqual(r, expected_r, places=4)

        # Resized width should be 416, resized height should be round(480 * 0.65) = 312
        # Vertical padding should be (416 - 312) / 2 = 52
        self.assertEqual(left, 0)
        self.assertEqual(top, 52)


class TestPreprocess(unittest.TestCase):
    def test_preprocess_tensor(self):
        # Create synthetic letterboxed image [416, 416, 3] with values 0..255
        img = np.full((416, 416, 3), 128, dtype=np.uint8)
        tensor = preprocess(img)

        # Expected shape: [1, 3, 416, 416]
        self.assertEqual(tensor.shape, (1, 3, 416, 416))
        self.assertEqual(tensor.dtype, np.float32)

        # Normalization check: 128 / 255 ≈ 0.50196
        self.assertTrue(np.all(tensor >= 0.0) and np.all(tensor <= 1.0))
        self.assertAlmostEqual(tensor[0, 0, 0, 0], 128.0 / 255.0, places=4)


class TestPostprocess(unittest.TestCase):
    def test_postprocess_synthetic_yolov5(self):
        # Synthetic output for 4 classes: 5 + 4 = 9 channels
        # Grid boxes count: 25200 (standard YOLOv5 anchor grid)
        num_boxes = 25200
        out = np.zeros((1, num_boxes, 9), dtype=np.float32)

        names = APPLE_CLASSES  # ["apple_fresh", "apple_rotten", "apple_bruised", "apple_damaged"]
        r = 0.65
        pad = (0, 52)
        orig_hw = (480, 640)

        # Inject box 1: healthy apple (apple_fresh = class 0)
        # Center at (208, 208) on 416x416 letterbox, w=100, h=100
        out[0, 100, 0] = 208.0   # cx
        out[0, 100, 1] = 208.0   # cy
        out[0, 100, 2] = 100.0   # w
        out[0, 100, 3] = 100.0   # h
        out[0, 100, 4] = 0.95    # obj conf
        out[0, 100, 5] = 0.90    # class 0: apple_fresh

        # Inject box 2: rotten defect (apple_rotten = class 1)
        # Center at (300, 300) on 416x416 letterbox, w=80, h=80
        out[0, 500, 0] = 300.0   # cx
        out[0, 500, 1] = 300.0   # cy
        out[0, 500, 2] = 80.0    # w
        out[0, 500, 3] = 80.0    # h
        out[0, 500, 4] = 0.92    # obj conf
        out[0, 500, 6] = 0.95    # class 1: apple_rotten

        dets = postprocess(out, names, r, pad, orig_hw, conf_t=0.25, iou_t=0.45)

        self.assertEqual(len(dets), 2)
        det_names = {d["name"] for d in dets}
        self.assertIn("apple_fresh", det_names)
        self.assertIn("apple_rotten", det_names)

        # Check coordinate unpadding back to orig 640x480 space
        for d in dets:
            x1, y1, x2, y2 = d["box"]
            self.assertTrue(0 <= x1 < x2 <= 640)
            self.assertTrue(0 <= y1 < y2 <= 480)
            self.assertTrue(d["conf"] > 0.25)


class TestVerdictLogic(unittest.TestCase):
    def test_is_defect(self):
        # FRESH verdicts
        self.assertFalse(is_defect("apple_fresh"))
        self.assertFalse(is_defect("mango_fresh"))
        self.assertFalse(is_defect("guava_fresh"))

        # DEFECT verdicts
        self.assertTrue(is_defect("apple_rotten"))
        self.assertTrue(is_defect("apple_bruised"))
        self.assertTrue(is_defect("apple_damaged"))
        self.assertTrue(is_defect("guava_rotten"))
        self.assertTrue(is_defect("guava_bruised"))
        self.assertTrue(is_defect("guava_damaged"))
        self.assertTrue(is_defect("crack"))


class TestLoadNames(unittest.TestCase):
    def test_load_names_cli(self):
        cli = "apple_fresh, apple_rotten, apple_bruised, apple_damaged"
        names = load_names(None, cli, APPLE_CLASSES)
        self.assertEqual(names, APPLE_CLASSES)

    def test_load_names_metadata_dict(self):
        mock_sess = MagicMock()
        meta = MagicMock()
        meta.custom_metadata_map = {
            "names": "{0: 'apple_fresh', 1: 'apple_rotten', 2: 'apple_bruised', 3: 'apple_damaged'}"
        }
        mock_sess.get_modelmeta.return_value = meta

        names = load_names(mock_sess, None, APPLE_CLASSES)
        self.assertEqual(names, APPLE_CLASSES)

    def test_load_names_metadata_list(self):
        mock_sess = MagicMock()
        meta = MagicMock()
        meta.custom_metadata_map = {
            "names": "['apple_fresh', 'apple_rotten', 'apple_bruised', 'apple_damaged']"
        }
        mock_sess.get_modelmeta.return_value = meta

        names = load_names(mock_sess, None, APPLE_CLASSES)
        self.assertEqual(names, APPLE_CLASSES)

    def test_load_names_mismatch_raises(self):
        cli = "apple_fresh, apple_rotten, other_defect"
        with self.assertRaises(SystemExit):
            load_names(None, cli, APPLE_CLASSES)


class TestDatasetRemapping(unittest.TestCase):
    def test_unified_classes_definition(self):
        self.assertEqual(len(UNIFIED_CLASSES), 4)
        self.assertEqual(UNIFIED_CLASSES[0], "apple_fresh")
        self.assertEqual(UNIFIED_CLASSES[1], "apple_rotten")
        self.assertEqual(UNIFIED_CLASSES[2], "apple_bruised")
        self.assertEqual(UNIFIED_CLASSES[3], "apple_damaged")

    def test_class_remapping_rules(self):
        # 0: apple_fresh
        self.assertEqual(remap_class("HEALTHY"), 0)
        self.assertEqual(remap_class("healthy"), 0)

        # 1: apple_rotten
        self.assertEqual(remap_class("ROT"), 1)
        self.assertEqual(remap_class("rotten"), 1)

        # 2: apple_bruised
        self.assertEqual(remap_class("BLOTCH"), 2)
        self.assertEqual(remap_class("bruising"), 2)
        self.assertEqual(remap_class("bruised"), 2)

        # 3: apple_damaged
        self.assertEqual(remap_class("SCAB"), 3)
        self.assertEqual(remap_class("crack"), 3)
        self.assertEqual(remap_class("insect_bite"), 3)
        self.assertEqual(remap_class("damaged"), 3)

        # Dropped classes
        self.assertIsNone(remap_class("russetting"))
        self.assertIsNone(remap_class("rusetting"))
        self.assertIsNone(remap_class("objects"))
        self.assertIsNone(remap_class("object"))
        self.assertIsNone(remap_class("unknown_label"))


class TestMangoDatasetRemapping(unittest.TestCase):
    def test_mango_classes_definition(self):
        import dataset_prep_mango
        self.assertEqual(len(dataset_prep_mango.UNIFIED_CLASSES), 4)
        self.assertEqual(dataset_prep_mango.UNIFIED_CLASSES, ["mango_fresh", "mango_rotten", "mango_bruised", "mango_damaged"])

    def test_mango_remapping_rules(self):
        import dataset_prep_mango
        # 0: fresh
        self.assertEqual(dataset_prep_mango.remap_class("Grade-A"), 0)
        self.assertEqual(dataset_prep_mango.remap_class("green"), 0)
        self.assertEqual(dataset_prep_mango.remap_class("ripe"), 0)

        # 1: rotten
        self.assertEqual(dataset_prep_mango.remap_class("Black Spot"), 1)
        self.assertEqual(dataset_prep_mango.remap_class("rotten"), 1)
        self.assertEqual(dataset_prep_mango.remap_class("Grade-C"), 1)

        # 2: bruised
        self.assertEqual(dataset_prep_mango.remap_class("Brown Spot"), 2)
        self.assertEqual(dataset_prep_mango.remap_class("skin damage"), 2)
        self.assertEqual(dataset_prep_mango.remap_class("Grade-B"), 2)

        # 3: damaged
        self.assertEqual(dataset_prep_mango.remap_class("Mango Scab"), 3)


class TestGuavaDatasetRemapping(unittest.TestCase):
    def test_guava_classes_definition(self):
        import dataset_prep_guava
        self.assertEqual(len(dataset_prep_guava.UNIFIED_CLASSES), 4)
        self.assertEqual(
            dataset_prep_guava.UNIFIED_CLASSES,
            ["guava_fresh", "guava_rotten", "guava_bruised", "guava_damaged"],
        )

    def test_guava_remapping_rules(self):
        import dataset_prep_guava
        # 0: fresh
        self.assertEqual(dataset_prep_guava.remap_class("healthy"), 0)
        self.assertEqual(dataset_prep_guava.remap_class("fresh"), 0)

        # 1: rotten
        self.assertEqual(dataset_prep_guava.remap_class("Phytopthora"), 1)
        self.assertEqual(dataset_prep_guava.remap_class("Styler and Root"), 1)
        self.assertEqual(dataset_prep_guava.remap_class("Anthracnose"), 1)
        self.assertEqual(dataset_prep_guava.remap_class("fruit fly"), 1)

        # 2: bruised
        self.assertEqual(dataset_prep_guava.remap_class("Brown Spot"), 2)
        self.assertEqual(dataset_prep_guava.remap_class("surface abrasion"), 2)
        self.assertEqual(dataset_prep_guava.remap_class("blemish"), 2)

        # 3: damaged
        self.assertEqual(dataset_prep_guava.remap_class("Scab"), 3)
        self.assertEqual(dataset_prep_guava.remap_class("scab"), 3)
        self.assertEqual(dataset_prep_guava.remap_class("damaged"), 3)


class TestPomegranateDatasetRemapping(unittest.TestCase):
    def test_pomegranate_classes_definition(self):
        import dataset_prep_pomegranate
        self.assertEqual(len(dataset_prep_pomegranate.UNIFIED_CLASSES), 4)
        self.assertEqual(
            dataset_prep_pomegranate.UNIFIED_CLASSES,
            ["pomegranate_fresh", "pomegranate_rotten", "pomegranate_bruised", "pomegranate_damaged"],
        )

    def test_pomegranate_remapping_rules(self):
        import dataset_prep_pomegranate
        # 0: fresh
        self.assertEqual(dataset_prep_pomegranate.remap_class("healthy"), 0)
        self.assertEqual(dataset_prep_pomegranate.remap_class("Fresh Pomegranate"), 0)
        self.assertEqual(dataset_prep_pomegranate.remap_class("freshpomegranate"), 0)

        # 1: rotten
        self.assertEqual(dataset_prep_pomegranate.remap_class("Alternaria"), 1)
        self.assertEqual(dataset_prep_pomegranate.remap_class("Anthracnose"), 1)
        self.assertEqual(dataset_prep_pomegranate.remap_class("Spoiled Pomegranate"), 1)

        # 2: bruised
        self.assertEqual(dataset_prep_pomegranate.remap_class("Cercospora"), 2)
        self.assertEqual(dataset_prep_pomegranate.remap_class("surface lesions"), 2)

        # 3: damaged
        self.assertEqual(dataset_prep_pomegranate.remap_class("Bacterial_Blight"), 3)
        self.assertEqual(dataset_prep_pomegranate.remap_class("cracks"), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
