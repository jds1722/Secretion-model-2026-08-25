from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile

from augment_training_images import (
    Augmentation,
    AugmentationOptions,
    apply_augmentation,
    augment_images,
    read_tiff,
)


class ArrayTransformTests(unittest.TestCase):
    def test_yxs_flip_changes_x_not_samples(self) -> None:
        image = np.arange(3 * 4 * 3, dtype=np.uint16).reshape(3, 4, 3)
        result = apply_augmentation(image, "YXS", Augmentation("flip", flip_lr=True))
        np.testing.assert_array_equal(result[:, 0, :], image[:, -1, :])
        np.testing.assert_array_equal(result[:, :, 0], image[:, ::-1, 0])

    def test_integer_translation_does_not_create_intensities(self) -> None:
        image = np.arange(30, dtype=np.uint16).reshape(5, 6)
        result = apply_augmentation(
            image, "YX", Augmentation("shift", shift_y=1, shift_x=-2), fill_value=0
        )
        self.assertTrue(set(np.unique(result)).issubset(set(np.unique(image))))
        np.testing.assert_array_equal(result[1:, :-2], image[:-1, 2:])

    def test_nearest_rotation_does_not_create_intensities(self) -> None:
        image = np.arange(49, dtype=np.uint16).reshape(7, 7)
        result = apply_augmentation(
            image, "YX", Augmentation("small", angle_degrees=5), fill_value=0
        )
        self.assertTrue(set(np.unique(result)).issubset(set(np.unique(image))))

    def test_border_median_fill_is_independent_for_each_plane(self) -> None:
        image = np.empty((2, 7, 7), dtype=np.uint16)
        image[0] = 100
        image[1] = 1000
        image[0, 2:5, 2:5] = 500
        image[1, 2:5, 2:5] = 5000
        result = apply_augmentation(
            image,
            "QYX",
            Augmentation("shift", shift_x=2),
            fill_mode="border_median",
        )
        np.testing.assert_array_equal(result[0, :, :2], 100)
        np.testing.assert_array_equal(result[1, :, :2], 1000)
        self.assertTrue(set(np.unique(result)).issubset(set(np.unique(image))))


class TiffWorkflowTests(unittest.TestCase):
    def test_ome_axes_channel_names_and_physical_sizes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.ome.tif"
            output = root / "output"
            data = np.arange(2 * 4 * 5, dtype=np.uint16).reshape(2, 4, 5)
            tifffile.imwrite(
                source,
                data,
                ome=True,
                photometric="minisblack",
                metadata={
                    "axes": "CYX",
                    "PhysicalSizeX": 0.2,
                    "PhysicalSizeXUnit": "µm",
                    "PhysicalSizeY": 0.3,
                    "PhysicalSizeYUnit": "µm",
                    "Channel": {"Name": ["nucleus", "membrane"]},
                },
            )
            options = AugmentationOptions(
                include_quarter_turns=True,
                include_horizontal_flips=False,
            )
            results, _ = augment_images(source, output, False, False, options)
            self.assertTrue(all(result.status == "written" for result in results))
            rotated = read_tiff(output / "images" / "source.ome_rot90.tif")
            self.assertEqual(rotated.axes, "CYX")
            self.assertEqual(rotated.data.shape, (2, 5, 4))
            self.assertEqual(rotated.metadata["PhysicalSizeX"], 0.3)
            self.assertEqual(rotated.metadata["PhysicalSizeY"], 0.2)
            self.assertEqual(
                rotated.metadata["Channel"]["Name"], ["nucleus", "membrane"]
            )

    def test_paths_masks_and_resolution_are_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            images = root / "images"
            masks = root / "masks"
            output = root / "output"
            for group, offset in (("a", 0), ("b", 100)):
                (images / group).mkdir(parents=True)
                (masks / group).mkdir(parents=True)
                image = np.arange(20, dtype=np.uint16).reshape(4, 5) + offset
                mask = (image % 3 == 0).astype(np.uint8)
                tifffile.imwrite(
                    images / group / "sample.tif",
                    image,
                    metadata={"axes": "YX"},
                    resolution=(2, 3),
                    resolutionunit="CENTIMETER",
                )
                tifffile.imwrite(masks / group / "sample.tif", mask, metadata={"axes": "YX"})

            options = AugmentationOptions(
                include_quarter_turns=True,
                include_horizontal_flips=False,
            )
            results, report = augment_images(images, output, True, False, options, masks)
            self.assertEqual(len(results), 8)
            self.assertTrue(all(result.status == "written" for result in results))
            self.assertTrue(report.exists())
            for group in ("a", "b"):
                self.assertTrue((output / "images" / group / "sample_orig.tif").exists())
                self.assertTrue((output / "masks" / group / "sample_orig.tif").exists())
                original = tifffile.imread(images / group / "sample.tif")
                copied = tifffile.imread(output / "images" / group / "sample_orig.tif")
                copied_mask = tifffile.imread(output / "masks" / group / "sample_orig.tif")
                np.testing.assert_array_equal(copied, original)
                np.testing.assert_array_equal(
                    copied_mask, tifffile.imread(masks / group / "sample.tif")
                )
                with tifffile.TiffFile(output / "images" / group / "sample_orig.tif") as tif:
                    self.assertEqual(tif.pages[0].tags["XResolution"].value, (2, 1))
                    self.assertEqual(tif.pages[0].tags["YResolution"].value, (3, 1))
                with tifffile.TiffFile(output / "images" / group / "sample_rot90.tif") as tif:
                    self.assertEqual(tif.pages[0].tags["XResolution"].value, (3, 1))
                    self.assertEqual(tif.pages[0].tags["YResolution"].value, (2, 1))

    def test_output_inside_input_is_allowed_and_excluded_from_recursive_input(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir) / "input"
            input_dir.mkdir()
            tifffile.imwrite(input_dir / "sample.tif", np.zeros((3, 3), dtype=np.uint8))
            output_dir = input_dir / "output"
            (output_dir / "old").mkdir(parents=True)
            tifffile.imwrite(
                output_dir / "old" / "previous_result.tif",
                np.ones((3, 3), dtype=np.uint8),
            )
            options = AugmentationOptions(
                include_quarter_turns=False,
                include_horizontal_flips=False,
            )
            first_results, _ = augment_images(
                input_dir, output_dir, True, False, options
            )
            second_results, _ = augment_images(
                input_dir, output_dir, True, False, options
            )
            self.assertEqual(len(first_results), 1)
            self.assertEqual(first_results[0].status, "written")
            self.assertEqual(len(second_results), 1)
            self.assertEqual(second_results[0].status, "skipped")

    def test_output_equal_to_input_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir) / "input"
            input_dir.mkdir()
            tifffile.imwrite(input_dir / "sample.tif", np.zeros((3, 3), dtype=np.uint8))
            with self.assertRaisesRegex(ValueError, "same as"):
                augment_images(input_dir, input_dir, False, False)


if __name__ == "__main__":
    unittest.main()
