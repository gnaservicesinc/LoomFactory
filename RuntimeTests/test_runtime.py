import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Runtime"))
from loom_factory.engine import Cancellation, TrainingCancelled, train, validate_request
from loom_factory.images import exaggerate_depth, native_pair, read_target, read_rgb
from loom_factory.models import MaterialMapNetwork
from loom_factory.metadata import write_fixture_metadata


class PixelContractTests(unittest.TestCase):
    def test_native_sixteen_bit_rgb_png_and_tiff_do_not_pass_through_eight_bit(self):
        import png
        import tifffile
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            values = np.array([[[1, 1001, 65535], [12345, 32769, 65001]],
                               [[5555, 44444, 11111], [256, 257, 258]]], dtype=np.uint16)
            png_path = root / "rgb16.png"
            with png_path.open("wb") as stream:
                png.Writer(2, 2, greyscale=False, bitdepth=16).write(stream, values.reshape(2, 6))
            tiff_path = root / "rgb16.tiff"
            tifffile.imwrite(tiff_path, values, photometric="rgb")
            for path in (png_path, tiff_path):
                np.testing.assert_allclose(read_rgb(str(path)), values / 65535.0, atol=1e-7)
                np.testing.assert_allclose(read_target(str(path), "normal"), values / 65535.0, atol=1e-7)

    def test_exr_float_scalar_and_heic_input_have_executable_decoders(self):
        import OpenEXR
        import pillow_heif
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            values = np.array([[0.2, 0.4], [0.6, 0.8]], dtype=np.float32)
            with OpenEXR.File({}, {"Y": values}) as file:
                file.write(str(root / "depth.exr"))
            np.testing.assert_allclose(read_target(str(root / "depth.exr"), "depth")[..., 0], values)
            rgb = np.tile(np.array([90, 140, 210], dtype=np.uint8), (32, 32, 1))
            pillow_heif.from_pillow(Image.fromarray(rgb)).save(str(root / "input.heic"), quality=100)
            decoded = read_rgb(str(root / "input.heic"))
            self.assertEqual(decoded.shape, (32, 32, 3))
            self.assertTrue(np.isfinite(decoded).all())
            np.testing.assert_allclose(decoded, rgb / 255.0, atol=3 / 255.0)

    def test_sixteen_bit_targets_use_code_range_not_observed_range(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "target.png"
            values = np.array([[20000, 30000], [40000, 50000]], dtype=np.uint16)
            Image.fromarray(values).save(path)
            result = read_target(str(path), "depth")
            np.testing.assert_allclose(result[..., 0], values / 65535.0, rtol=1e-6)
            self.assertGreater(float(result.min()), 0)
            self.assertLess(float(result.max()), 1)

    def test_pair_crops_share_exact_native_pixels_and_sources_are_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ramp = np.arange(48 * 48, dtype=np.uint16).reshape(48, 48) % 256
            Image.fromarray(np.repeat(ramp[..., None].astype(np.uint8), 3, axis=2)).save(root / "rgb.png")
            Image.fromarray(ramp * 257).save(root / "depth.png")
            hashes = [hashlib.sha256(path.read_bytes()).hexdigest() for path in (root / "rgb.png", root / "depth.png")]
            sample = {"id": "paired", "input": str(root / "rgb.png"), "target": str(root / "depth.png")}
            rgb, target, crop = native_pair(sample, 32, random.Random(7), "depth", True, True, False)
            self.assertEqual(tuple(rgb.shape), (3, 32, 32))
            torch.testing.assert_close(rgb[0], target[0])
            self.assertTrue(crop["x"] or crop["y"])
            self.assertEqual(hashes, [hashlib.sha256(path.read_bytes()).hexdigest() for path in (root / "rgb.png", root / "depth.png")])

    def test_exaggeration_preserves_endpoints_midpoint_and_constant_depth(self):
        value = torch.tensor([0.2, 0.35, 0.5, 0.65, 0.8])
        result = exaggerate_depth(value)
        self.assertAlmostEqual(float(result[0]), 0.2)
        self.assertAlmostEqual(float(result[-1]), 0.8)
        self.assertAlmostEqual(float(result[2]), 0.5)
        self.assertLess(float(result[1]), 0.35)
        self.assertGreater(float(result[3]), 0.65)
        torch.testing.assert_close(exaggerate_depth(torch.ones(5) * 0.3), torch.ones(5) * 0.3)

    def test_directx_targets_are_converted_once_to_opengl(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "normal.png"
            Image.fromarray(np.tile(np.array([128, 32, 255], dtype=np.uint8), (4, 4, 1))).save(path)
            gl = read_target(str(path), "normal", "OpenGL")
            dx = read_target(str(path), "normal", "DirectX")
            np.testing.assert_allclose(dx[..., 1], 1 - gl[..., 1])
            np.testing.assert_allclose(dx[..., (0, 2)], gl[..., (0, 2)])


class RequestMembershipTests(unittest.TestCase):
    def test_external_path_is_rejected_but_a_present_source_symlink_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            external = root / "actual.png"
            Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(external)
            request = {"schema": "loom-factory-training-v1", "project_id": "p", "dataset_id": "d",
                       "source_root": str(source), "output_directory": str(source / ".loom-factory" / "runs" / "a"),
                       "map_kind": "depth", "image_size": 32,
                       "samples": [{"id": "a", "input": str(external), "target": str(external)}]}
            with self.assertRaisesRegex(ValueError, "outside"):
                validate_request(request)
            linked = source / "linked.png"
            linked.symlink_to(external)
            request["samples"] = [{"id": "a", "input": str(linked), "target": str(linked)}]
            self.assertEqual(len(validate_request(request)[2]), 1)
            linked.unlink()
            with self.assertRaisesRegex(ValueError, "moved or deleted"):
                validate_request(request)
            request["samples"] = [{"id": "a", "input": str(source / ".." / "actual.png"), "target": str(external)}]
            with self.assertRaisesRegex(ValueError, "outside"):
                validate_request(request)

    def test_outputs_cannot_escape_project_storage_via_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            external = root / "outside"
            external.mkdir()
            (source / ".loom-factory").symlink_to(external)
            request = {"schema": "loom-factory-training-v1", "source_root": str(source),
                       "output_directory": str(source / ".loom-factory" / "run")}
            with self.assertRaisesRegex(ValueError, "inside this source"):
                validate_request(request)


class RealTrainingTests(unittest.TestCase):
    def make_samples(self, root, count=3):
        result = []
        for index in range(count):
            rgb = np.random.default_rng(index).integers(0, 256, (32, 32, 3), dtype=np.uint8)
            target = (rgb[..., 0].astype(np.uint16) * 257)
            rgb_path, target_path = root / f"{index}_rgb.png", root / f"{index}_depth.png"
            Image.fromarray(rgb).save(rgb_path)
            Image.fromarray(target).save(target_path)
            result.append({"id": str(index), "input": str(rgb_path), "target": str(target_path)})
        return result

    def request(self, root, samples, run):
        return {"schema": "loom-factory-training-v1", "project_id": "project", "dataset_id": "bricks",
                "dataset_name": "Bricks", "source_root": str(root), "output_directory": str(root / ".loom-factory" / "runs" / run),
                "map_kind": "depth", "image_size": 32, "width": 8, "blocks": 1,
                "epochs": 2, "batch_size": 1, "shuffle_each_epoch": False,
                "flip_crops": False, "rotate_crops": False, "exaggerate_features": True, "device": "cpu", "samples": samples}

    def test_actual_training_fixed_order_safetensors_and_flexible_coreml_prediction(self):
        import coremltools as ct
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = self.request(root, self.make_samples(root), "map")
            with contextlib.redirect_stdout(io.StringIO()) as output:
                result = train(request)
            events = [json.loads(line) for line in output.getvalue().splitlines()]
            self.assertEqual([e["sample_ids"] for e in events if e["event"] == "progress"], [["0"], ["1"], ["2"], ["0"], ["1"], ["2"]])
            history = result["training"]["history"]
            self.assertEqual(history[0]["order_sha256"], history[1]["order_sha256"])
            self.assertEqual(result["training"]["updates"], 6)
            self.assertEqual(result["training"]["decoded_image_cache"]["hits"], 6)
            self.assertEqual(result["training"]["decoded_image_cache"]["misses"], 6)
            self.assertEqual(result["training"]["depth_transform"], "anchored-tanh-contrast-strength-2")
            run = Path(request["output_directory"])
            model = MaterialMapNetwork("depth", 8, 1).eval()
            model.load_state_dict(load_file(str(run / "model.safetensors")))
            rgb = np.random.default_rng(42).random((1, 3, 48, 48), dtype=np.float32)
            expected = model(torch.from_numpy(rgb)).detach().numpy()
            deployed = ct.models.MLModel(str(run / "model.mlpackage"), compute_units=ct.ComputeUnit.CPU_ONLY)
            actual = deployed.predict({"rgb": rgb})["material_map"]
            np.testing.assert_allclose(actual, expected, rtol=3e-4, atol=3e-4)
            self.assertEqual(tuple(actual.shape), (1, 1, 48, 48))

    def test_classifier_training_command_is_unavailable(self):
        runtime = Path(__file__).resolve().parents[1] / "Runtime"
        result = subprocess.run(
            [sys.executable, "-m", "loom_factory", "train-classifier", "--request", "missing.json"],
            env={**os.environ, "PYTHONPATH": str(runtime)}, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("invalid choice", result.stderr)
        self.assertIn("train-map", result.stderr)

    def test_normal_training_and_native_coreml_match_unit_vector_output(self):
        import coremltools as ct
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            samples = self.make_samples(root, count=1)
            normal = np.tile(np.array([128, 128, 255], dtype=np.uint8), (32, 32, 1))
            Image.fromarray(normal).save(samples[0]["target"])
            request = self.request(root, samples, "normal")
            request["map_kind"] = "normal"
            request["epochs"] = 1
            request["exaggerate_features"] = False
            with contextlib.redirect_stdout(io.StringIO()):
                result = train(request)
            self.assertEqual(result["architecture_config"]["normal_convention"], "OpenGL")
            output = Path(request["output_directory"])
            reference = MaterialMapNetwork("normal", 8, 1).eval()
            reference.load_state_dict(load_file(str(output / "model.safetensors")))
            rgb = np.random.default_rng(8).random((1, 3, 48, 48), dtype=np.float32)
            expected = reference(torch.from_numpy(rgb)).detach().numpy()
            deployed = ct.models.MLModel(str(output / "model.mlpackage"), compute_units=ct.ComputeUnit.CPU_ONLY)
            actual = deployed.predict({"rgb": rgb})["material_map"]
            np.testing.assert_allclose(actual, expected, rtol=3e-4, atol=3e-4)
            np.testing.assert_allclose(np.linalg.norm(actual * 2 - 1, axis=1), 1, atol=3e-4)

    def test_cancelled_run_does_not_publish_a_completed_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = self.request(root, self.make_samples(root), "cancel")
            cancellation = Cancellation()
            original_emit = __import__("loom_factory.engine", fromlist=["emit"]).emit
            def cancel_after_first_update(event, **fields):
                original_emit(event, **fields)
                if event == "progress":
                    cancellation.requested = True
            from unittest.mock import patch
            with contextlib.redirect_stdout(io.StringIO()), patch("loom_factory.engine.emit", cancel_after_first_update):
                with self.assertRaises(TrainingCancelled):
                    train(request, cancellation=cancellation)
            output = Path(request["output_directory"])
            self.assertFalse((output / "model.json").exists())
            self.assertEqual(json.loads((output / "progress.json").read_text())["status"], "cancelled")
            self.assertTrue((output / "checkpoint.safetensors").is_file())
            checkpoint = json.loads((output / "checkpoint.json").read_text())
            self.assertEqual(checkpoint["status"], "cancelled")
            self.assertEqual(checkpoint["updates"], 1)

    def test_weight_continuation_uses_current_members_and_validates_owner_and_architecture(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = self.request(root, self.make_samples(root, count=1), "first")
            first["epochs"] = 1
            with contextlib.redirect_stdout(io.StringIO()):
                train(first)
            checkpoint = Path(first["output_directory"]) / "checkpoint.safetensors"
            original_weights = load_file(str(checkpoint))
            # Add current physical members. The checkpoint never supplies data membership.
            second = self.request(root, self.make_samples(root, count=2), "continued")
            second["epochs"] = 1
            second["resume_weights"] = str(checkpoint)
            with contextlib.redirect_stdout(io.StringIO()):
                result = train(second)
            self.assertEqual(result["training"]["updates"], 2)
            continued = load_file(str(Path(second["output_directory"]) / "model.safetensors"))
            self.assertTrue(any(not torch.equal(original_weights[key], continued[key]) for key in original_weights))
            for field, value, reason in (("dataset_id", "wood", "different project, dataset"),
                                         ("width", 16, "same network width"),
                                         ("image_size", 64, "same native crop"),
                                         ("exaggerate_features", False, "same Exaggerate")):
                altered = {**second, field: value, "output_directory": str(root / ".loom-factory" / "runs" / field)}
                with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, reason):
                    train(altered)

    def test_live_tag_change_stops_before_a_retagged_image_is_used(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            samples = self.make_samples(root, count=2)
            category_id = "dataset-" + hashlib.sha256(b"bricks").hexdigest()[:16]
            for sample in samples:
                for field, role in (("input", "diffuse"), ("target", "depth")):
                    write_fixture_metadata(sample[field], {"schema": "loom-factory.image.v1", "id": field + sample["id"],
                        "pairingKey": sample["id"], "role": role, "category": "Bricks", "verified": True, "note": ""})
            request = self.request(root, samples, "retag")
            request["dataset_id"] = category_id
            request["require_image_metadata"] = True
            events = []
            def retag_next_image(event, **fields):
                events.append({"event": event, **fields})
                if event == "progress":
                    write_fixture_metadata(samples[1]["input"], {"schema": "loom-factory.image.v1", "id": "input1",
                        "pairingKey": "1", "role": "diffuse", "category": "Wood", "verified": True, "note": ""})
            from unittest.mock import patch
            with patch("loom_factory.engine.emit", retag_next_image):
                with self.assertRaisesRegex(ValueError, "dataset tag"):
                    train(request)
            self.assertEqual([e["sample_ids"] for e in events if e["event"] == "progress"], [["0"]])
            self.assertFalse((Path(request["output_directory"]) / "model.json").exists())

    def test_shuffled_epochs_differ_for_the_fixture(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = self.request(root, self.make_samples(root, count=5), "shuffle")
            request["shuffle_each_epoch"] = True
            with contextlib.redirect_stdout(io.StringIO()):
                result = train(request)
            self.assertNotEqual(result["training"]["history"][0]["order_sha256"], result["training"]["history"][1]["order_sha256"])

    def test_cached_pixels_cannot_bypass_a_changed_tag_next_epoch(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            samples = self.make_samples(root, count=1)
            tags = {}
            for field, role in (("input", "diffuse"), ("target", "depth")):
                tags[field] = {"schema": "loom-factory.image.v1", "id": field,
                               "pairingKey": "0", "role": role, "category": "Bricks", "verified": True}
                write_fixture_metadata(samples[0][field], tags[field])
            request = self.request(root, samples, "cached-retag")
            request.update(dataset_id="dataset-" + hashlib.sha256(b"bricks").hexdigest()[:16], require_image_metadata=True)
            updates = []
            def retag_after_first_update(event, **fields):
                if event == "progress":
                    updates.append(fields["update"])
                    write_fixture_metadata(samples[0]["input"], {**tags["input"], "category": "Wood"})
            with patch("loom_factory.engine.emit", retag_after_first_update):
                with self.assertRaisesRegex(ValueError, "dataset tag"):
                    train(request)
            self.assertEqual(updates, [1])
            self.assertFalse((Path(request["output_directory"]) / "model.json").exists())

    def test_non_finite_loss_never_updates_weights_or_exports(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = self.request(root, self.make_samples(root, count=1), "nonfinite")
            def nonfinite_loss(prediction, target, kind):
                return prediction.mean() * float("nan")
            with contextlib.redirect_stdout(io.StringIO()), patch("loom_factory.engine.map_loss", nonfinite_loss):
                with self.assertRaisesRegex(ValueError, "non-finite loss"):
                    train(request)
            output = Path(request["output_directory"])
            self.assertFalse((output / "model.json").exists())
            self.assertFalse((output / "checkpoint.safetensors").exists())
            self.assertEqual(json.loads((output / "progress.json").read_text())["updates"], 0)


if __name__ == "__main__":
    unittest.main()
