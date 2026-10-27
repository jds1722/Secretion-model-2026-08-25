import unittest

import pandas as pd

from src.training_manifest import assign_manifest_splits, is_original_variant, source_id


def samples(high_sizes, low_sizes, hormone="insulin"):
    return pd.DataFrame([
        {"hormone": hormone, "label": label, "source_id": f"{label}_{i}",
         "path": f"{hormone}_{label}_{i}_variant_{j}.tif"}
        for label, sizes in (("high", high_sizes), ("low", low_sizes))
        for i, size in enumerate(sizes)
        for j in range(size)
    ])


class BalancedTestSplitTests(unittest.TestCase):
    def check_invariants(self, original, result):
        self.assertEqual(sorted(original.path), sorted(result.path))
        self.assertTrue(result.groupby(["hormone", "label", "source_id"]).split.nunique().eq(1).all())
        for _, rows in result.groupby("hormone"):
            counts = rows[rows.split == "test"].groupby("label").size()
            self.assertEqual(counts["high"], counts["low"])
            for _, group in rows.groupby("label"):
                self.assertEqual(set(group.split), {"train", "val", "test"})

    def test_smaller_class_budget_and_extra_images_in_training(self):
        original = samples([1] * 100, [1] * 200)
        result = assign_manifest_splits(original, 0.15, 0.15, 17)
        self.check_invariants(original, result)
        counts = result.groupby(["label", "split"]).size()
        self.assertEqual(counts["high", "test"], 15)
        self.assertEqual(counts["low", "test"], 15)
        self.assertEqual(counts["high", "val"], 15)
        self.assertEqual(counts["low", "val"], 30)
        self.assertEqual(counts["low", "train"], 155)

    def test_rounds_down_without_splitting_augmented_sources(self):
        original = samples([8] * 25, [8] * 40)
        result = assign_manifest_splits(original, 0.15, 0.15, 17)
        self.check_invariants(original, result)
        # Budget is 30 images; whole groups permit 24 per class.
        self.assertEqual(len(result[(result.label == "high") & (result.split == "test")]), 24)

    def test_mixed_group_sizes_and_multiple_hormones_are_deterministic(self):
        original = pd.concat([
            samples([3, 5, 8, 2, 4] * 10, [4, 6, 1, 7] * 20),
            samples([8] * 25, [8] * 40, "glucagon"),
        ], ignore_index=True)
        result = assign_manifest_splits(original, 0.15, 0.15, 17)
        self.check_invariants(original, result)
        reordered = assign_manifest_splits(original.sample(frac=1, random_state=3), 0.15, 0.15, 17)
        self.assertEqual(result.set_index("path").split.to_dict(), reordered.set_index("path").split.to_dict())

    def test_impossible_balance_fails_instead_of_leaking_sources(self):
        with self.assertRaisesRegex(ValueError, "cannot form equal"):
            assign_manifest_splits(samples([8] * 3, [8] * 3), 0.15, 0.15, 17)

    def test_shift_rotation_and_flip_share_source_and_are_not_originals(self):
        from scripts.make_manifest import source_id as cli_source_id
        base = "Ar10_N17__Ar10_N17_1001_1"
        for direction, pixels in (
            (direction, pixels)
            for direction in ("up", "down", "left", "right")
            for pixels in (5, 10, 20)
        ):
            shift = f"_shift_{direction}_{pixels}px"
            for suffix in (shift, "_orig" + shift, "_rot90_flip_lr" + shift,
                           shift + "_rot270_flip_lr", shift + "_flip_lr"):
                for directory in ("/data/", "C:\\data\\"):
                    path = directory + base + suffix + ".tif"
                    with self.subTest(path=path):
                        self.assertEqual(source_id(path), base)
                        self.assertEqual(cli_source_id(path), base)
                        self.assertFalse(is_original_variant(path))
        self.assertTrue(is_original_variant(base + ".tif"))
        self.assertTrue(is_original_variant(base + "_orig.tif"))
        self.assertEqual(source_id(base + "_orig.tif"), base)
        self.assertNotEqual(source_id(base + "_orig.tif"), source_id(base[:-2] + "_orig.tif"))

    def test_shift_family_stays_together_in_balanced_splits(self):
        variants = ["_orig", "_orig_flip_lr", "_rot90", "_rot90_flip_lr",
                    "_rot180", "_rot180_flip_lr", "_rot270", "_rot270_flip_lr",
                    "_shift_up_5px", "_shift_down_5px", "_shift_left_5px", "_shift_right_5px",
                    "_shift_up_10px", "_shift_down_10px", "_shift_left_10px", "_shift_right_10px"]
        original = pd.DataFrame([
            {"hormone": "insulin", "label": label, "path": f"cell_{label}_{i}{suffix}.tif"}
            for label, count in (("high", 40), ("low", 80))
            for i in range(count) for suffix in variants
        ])
        original["source_id"] = original.path.map(source_id)
        result = assign_manifest_splits(original, 0.15, 0.15, 17)
        self.check_invariants(original, result)
        self.assertEqual(result.source_id.nunique(), 120)
        self.assertTrue(result.groupby("source_id").size().eq(16).all())
        self.assertEqual(len(result[(result.label == "high") & (result.split == "test")]), 96)
        previews = result[(result.split == "test") & result.path.map(is_original_variant)]
        self.assertEqual(len(previews), 12)


if __name__ == "__main__":
    unittest.main()
