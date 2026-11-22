import ast
import glob
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


def load_pairing():
    """Load the production function without importing model and GPU dependencies."""
    source = Path(__file__).resolve().parents[1] / "test_demo.py"
    tree = ast.parse(source.read_bytes(), filename=str(source))
    (function,) = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "select_dataset"]
    namespace = {"os": os, "glob": glob}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)  # noqa: S102
    return namespace["select_dataset"]


select_dataset = load_pairing()


class DatasetPairingTests(unittest.TestCase):
    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def assert_pairs(self, mode, root_name="dataset", filenames=(("000001.png", "000001x4.png"),)):
        root = self.root / root_name
        hr_dir = root / f"DIV2K_LSDIR_{mode}_HR"
        lr_dir = root / f"DIV2K_LSDIR_{mode}_LR"
        hr_dir.mkdir(parents=True)
        lr_dir.mkdir()
        expected = []
        for hr_name, lr_name in filenames:
            hr, lr = hr_dir / hr_name, lr_dir / lr_name
            hr.write_bytes(b"HR fixture")
            lr.write_bytes(b"LR fixture")
            expected.append((str(lr), str(hr)))

        pairs = select_dataset(root, mode)
        self.assertEqual(pairs, sorted(expected, key=lambda pair: pair[1]))
        self.assertEqual(len(pairs), len(set(pairs)))
        for lr, hr in pairs:
            self.assertEqual(Path(lr).read_bytes(), b"LR fixture")
            self.assertEqual(Path(hr).read_bytes(), b"HR fixture")
            self.assertEqual(Path(lr).parent.parent, root)
            self.assertEqual(Path(hr).parent.parent, root)

    def test_regular_validation_pair(self):
        self.assert_pairs("valid")

    def test_regular_test_pair(self):
        self.assert_pairs("test")

    def test_hr_in_root(self):
        self.assert_pairs("valid", "archive_HR")

    def test_png_in_root(self):
        self.assert_pairs("test", "archive.png")

    def test_both_suffixes_in_root(self):
        self.assert_pairs("valid", "archive_HR.png")

    def test_nested_hr_components(self):
        self.assert_pairs("test", "year_HR/snapshot_HR/data")

    def test_nested_png_components(self):
        self.assert_pairs("valid", "year.png/snapshot.png/data")

    def test_spaces_and_unicode_in_root(self):
        self.assert_pairs("test", "set with spaces/資料")

    def test_hr_in_filename_stem(self):
        self.assert_pairs("test", filenames=(("camera_HR.png", "camera_HRx4.png"),))

    def test_png_in_filename_stem(self):
        self.assert_pairs("valid", filenames=(("scene.png.backup.png", "scene.png.backupx4.png"),))

    def test_long_filename_with_spaces(self):
        stem = "large landscape " + "0" * 48
        self.assert_pairs("test", filenames=((stem + ".png", stem + "x4.png"),))

    def test_multiple_pairs_are_sorted_by_hr_name(self):
        filenames = (("0009.png", "0009x4.png"), ("0001.png", "0001x4.png"), ("0010.png", "0010x4.png"))
        for mode in ("valid", "test"):
            with self.subTest(mode=mode):
                self.assert_pairs(mode, mode, filenames)

    def test_empty_hr_directory(self):
        (self.root / "DIV2K_LSDIR_test_HR").mkdir()
        (self.root / "DIV2K_LSDIR_test_LR").mkdir()
        self.assertEqual(select_dataset(self.root, "test"), [])

    def test_unsupported_mode(self):
        with self.assertRaisesRegex(NotImplementedError, "train is not implemented in select_dataset"):
            select_dataset(self.root, "train")
