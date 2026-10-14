"""Native Swift export contract, using small compressed PNG/xattr fixtures."""
import ctypes
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest
import zlib


class MapTypeExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.build = tempfile.TemporaryDirectory(prefix="loom-map-export-build-")
        cls.binary = Path(cls.build.name) / "export-map-types"
        script = Path(__file__).resolve().parents[1] / "Scripts/export-map-type-dataset.swift"
        subprocess.run(["xcrun", "swiftc", "-O", str(script), "-o", str(cls.binary)], check=True, capture_output=True)

    @classmethod
    def tearDownClass(cls):
        cls.build.cleanup()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="loom-map-export-test-")
        self.root = Path(self.temp.name).resolve()
        self.source = self.root / "source"
        self.output = self.root / "output"
        self.source.mkdir()

    def tearDown(self):
        self.temp.cleanup()

    def image(self, name, role, size=1024, **tags):
        file = self.source / name
        file.parent.mkdir(parents=True, exist_ok=True)
        width, height = size if isinstance(size, tuple) else (size, size)
        def chunk(kind, payload):
            return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload))
        header = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
        pixels = zlib.compress((b"\0" + bytes([128]) * width) * height)
        file.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", pixels) + chunk(b"IEND", b""))
        if role is not None:
            value = dict(schema="loom-factory.image.v1", id=name, role=role, verified=False, **tags)
            data = json.dumps(value).encode()
            lib = ctypes.CDLL(None, use_errno=True)
            write = lib.setxattr
            write.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32, ctypes.c_int]
            write.restype = ctypes.c_int
            buffer = ctypes.create_string_buffer(data)
            if write(bytes(file), b"com.gnaservicesinc.loomfactory.image.v1", buffer, len(data), 0, 0):
                raise OSError(ctypes.get_errno(), "fixture xattr failed")
        return file

    def export(self, *extra):
        result = subprocess.run([str(self.binary), "--source", str(self.source), "--output", str(self.output), *extra],
                                check=True, capture_output=True, text=True)
        return json.loads(result.stdout)

    def test_actual_dimensions_and_tags_select_one_of_each_type_per_folder(self):
        # Misleading filenames prove selection uses dimensions and stored roles.
        chosen = self.image("Bark/renamed-depth-2k.png", "depth", 1024)
        self.image("Bark/depth-1k.png", "depth", 2048)
        self.image("Bark/depth-4k.png", "depth", 4096)
        rough = self.image("Bark/anything.png", "roughness", 2048)
        self.image("Bark/anything-larger.png", "roughness", 4096)
        self.image("Bark/bump.png", "bump")
        self.image("Bark/color.png", "diffuse")
        self.image("Bark/normals.png", "normal")
        self.image("Stone/color.png", "diffuse", 512)
        self.image("Stone/color-tiny.png", "diffuse", 256)
        before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in self.source.rglob("*.png")}
        report = self.export("--dry-run")
        self.assertFalse(self.output.exists())
        self.assertEqual(report["class_counts"], dict(depth=1, diffuse=2, normal=1, rough=1))
        self.assertEqual(report["skipped_images"], dict(bump_map=1))
        entries = {entry["class"]: entry for entry in report["entries"] if entry["material_folder"] == "Bark"}
        self.assertEqual(entries["depth"]["source"], str(chosen))
        self.assertEqual(entries["rough"]["source"], str(rough))
        report = self.export()
        for entry in report["entries"]:
            link = self.output / entry["path"]
            self.assertTrue(link.is_symlink())
            self.assertEqual(link.resolve(), Path(entry["source"]))
        self.assertEqual(before, {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in self.source.rglob("*.png")})

    def test_ignored_unknown_untagged_and_directx_never_enter_classes(self):
        self.image("Bark/color.png", "diffuse")
        self.image("Bark/ignored.png", "depth", ignored=True)
        self.image("Bark/unknown.png", "unknown")
        self.image("Bark/no-tag-diffuse.png", None)
        self.image("Bark/normal.png", "normal", normalConvention="directx")
        self.image("Bark/Bark_NormalDX.png", "normal")
        report = self.export()
        self.assertEqual(report["exported_images"], 1)
        self.assertEqual(report["skipped_images"], dict(ignored=1, unknown_map_type=1, untagged=1, directx_normal=2))

    def test_rectangular_1k_is_preferred_over_square_2k(self):
        first = self.image("Wood/color-small.png", "diffuse", (1024, 2048))
        self.image("Wood/color-large.png", "diffuse", 2048)
        report = self.export()
        self.assertEqual(report["entries"][0]["source"], str(first))
        self.assertEqual(report["entries"][0]["resolution_priority"], 0)

    def test_repeat_updates_owned_links_and_preserves_unrelated_files(self):
        first = self.image("Bark/depth.png", "depth", 2048)
        report = self.export()
        path = self.output / report["entries"][0]["path"]
        untouched = self.output / "depth/user-data.png"
        untouched.write_bytes(b"keep this")
        self.assertEqual(self.export()["exported_images"], 1)
        better = self.image("Bark/new-depth.png", "depth", 1024)
        updated = self.export()
        self.assertEqual(updated["entries"][0]["path"], report["entries"][0]["path"])
        self.assertEqual(path.resolve(), better)
        better.unlink()
        self.assertEqual(self.export()["entries"][0]["source"], str(first))
        self.assertEqual(path.resolve(), first)
        self.image("Stone/color.png", "diffuse")
        first.unlink()
        cleaned = self.export()
        self.assertFalse(path.is_symlink())
        self.assertEqual(cleaned["class_counts"]["depth"], 0)
        self.assertEqual(untouched.read_bytes(), b"keep this")

    def test_collision_preserves_existing_file_and_repeat_creates_no_duplicate(self):
        self.image("Bark/color.png", "diffuse")
        plan = self.export("--dry-run")
        reserved = self.output / plan["entries"][0]["path"]
        reserved.parent.mkdir(parents=True)
        reserved.write_bytes(b"unrelated image")
        first = self.export()
        self.assertNotEqual(first["entries"][0]["path"], plan["entries"][0]["path"])
        again = self.export()
        self.assertEqual(first["entries"][0]["path"], again["entries"][0]["path"])
        self.assertEqual(reserved.read_bytes(), b"unrelated image")
        self.assertEqual(sum(p.is_symlink() for p in reserved.parent.iterdir()), 1)

    def test_output_inside_source_is_rejected_before_creating_links(self):
        self.image("Bark/color.png", "diffuse")
        nested = self.source / "dataset"
        result = subprocess.run([str(self.binary), "--source", str(self.source), "--output", str(nested)], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(nested.exists())


if __name__ == "__main__":
    unittest.main()
