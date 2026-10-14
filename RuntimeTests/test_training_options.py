import contextlib
import hashlib
import io
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from safetensors.torch import load_file
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Runtime"))
from loom_factory.augmentation import AugmentationPlan
from loom_factory.engine import Cancellation, TrainingCancelled, train, validate_request
from loom_factory.images import native_pair
from loom_factory.models import MaterialMapNetwork
from loom_factory.metadata import write_fixture_metadata
from loom_factory.training_options import TrainingOptions, Lion, learning_rate_for_step, make_optimizer


def fake_export(model, output, size):
    output.mkdir()
    (output / "fixture").write_text("Conversion omitted in isolated optimizer tests.")


class OptionTests(unittest.TestCase):
    def test_all_optimizer_choices_are_used_with_selected_controls(self):
        for name, expected in (("adamw", torch.optim.AdamW), ("adam", torch.optim.Adam),
                               ("sgd", torch.optim.SGD), ("lion", Lion)):
            with self.subTest(name=name):
                options = TrainingOptions.from_request({"optimizer": name, "weight_decay": .07,
                                                        "beta1": .8, "beta2": .97, "momentum": .75})
                parameter = torch.nn.Parameter(torch.tensor([1., -1.]))
                optimizer = make_optimizer([parameter], options)
                self.assertIsInstance(optimizer, expected)
                self.assertEqual(optimizer.param_groups[0]["weight_decay"], .07)
                if name == "sgd":
                    self.assertEqual(optimizer.param_groups[0]["momentum"], .75)
                else:
                    self.assertEqual(optimizer.param_groups[0]["betas"], (.8, .97))
                parameter.grad = torch.tensor([.2, -.4])
                optimizer.step()
                self.assertLess(float(parameter[0]), 1)
                self.assertGreater(float(parameter[1]), -1)

    def test_lion_matches_two_published_algorithm_steps(self):
        parameter = torch.nn.Parameter(torch.tensor([2., -3.]))
        optimizer = Lion([parameter], lr=.1, betas=(.5, .8), weight_decay=.2)
        expected, momentum = parameter.detach().clone(), torch.zeros(2)
        for gradient in (torch.tensor([.4, -.2]), torch.tensor([-.05, .2])):
            direction = (.5 * momentum + .5 * gradient).sign()
            expected = expected * .98 - .1 * direction
            momentum = .8 * momentum + .2 * gradient
            parameter.grad = gradient
            optimizer.step()
            torch.testing.assert_close(parameter, expected)
            torch.testing.assert_close(optimizer.state[parameter]["momentum"], momentum)

    def test_options_reject_nonfinite_and_invalid_controls(self):
        invalid = ({"optimizer": "guess"}, {"gradient_accumulation_steps": 0},
                   {"gradient_accumulation_steps": True}, {"gradient_clip_norm": -1},
                   {"weight_decay": float("nan")}, {"beta1": 1}, {"beta2": -1},
                   {"optimizer": "sgd", "momentum": float("inf")}, {"lr_schedule": "guess"},
                   {"warmup_epochs": 2, "epochs": 2}, {"minimum_lr_ratio": 1.1},
                   {"crop_refresh": "never"}, {"augmentation_refresh": "never"})
        for request in invalid:
            with self.subTest(request=request), self.assertRaises(ValueError):
                TrainingOptions.from_request(request)

    def test_warmup_and_cosine_are_indexed_by_optimizer_steps(self):
        options = TrainingOptions.from_request({"learning_rate": .1, "lr_schedule": "cosine",
                                                "warmup_epochs": 1, "epochs": 3, "minimum_lr_ratio": .2})
        rates = [learning_rate_for_step(options, step, 6, 2) for step in range(6)]
        self.assertAlmostEqual(rates[0], .05)
        self.assertAlmostEqual(rates[1], .1)
        self.assertAlmostEqual(rates[2], .1)
        self.assertAlmostEqual(rates[-1], .02)
        self.assertGreater(rates[3], rates[4])
        self.assertAlmostEqual(learning_rate_for_step(options, 2, 3, 2), .1)
        single = TrainingOptions.from_request({"learning_rate": .1, "lr_schedule": "cosine"})
        self.assertAlmostEqual(learning_rate_for_step(single, 0, 1, 1), .1)

    def test_inactive_optimizer_parameters_are_ignored_and_weight_decay_matches_ui(self):
        sgd = TrainingOptions.from_request({"optimizer": "sgd", "beta1": 1, "beta2": float("nan")})
        self.assertEqual((sgd.beta1, sgd.beta2), (.9, .999))
        adam = TrainingOptions.from_request({"optimizer": "adam", "momentum": float("inf")})
        self.assertEqual(adam.momentum, .9)
        for optimizer in ("adamw", "adam", "sgd", "lion"):
            self.assertEqual(TrainingOptions.from_request({"optimizer": optimizer, "weight_decay": 10}).weight_decay, 10)
        with self.assertRaises(ValueError):
            TrainingOptions.from_request({"weight_decay": 10.1})


class AugmentationTimingTests(unittest.TestCase):
    def test_run_epoch_and_visit_timing_and_shuffle_independence(self):
        for refresh in ("per_run", "per_epoch", "per_sample"):
            plan = AugmentationPlan(42, refresh, refresh)
            first = [rng.getrandbits(100) for rng in plan.generators("a", 0)]
            same = [rng.getrandbits(100) for rng in plan.generators("a", 0)]
            next_epoch = [rng.getrandbits(100) for rng in plan.generators("a", 1)]
            self.assertEqual(first == same, refresh != "per_sample")
            self.assertEqual(first == next_epoch, refresh == "per_run")
        for refresh in ("per_run", "per_epoch"):
            first, second = AugmentationPlan(11, refresh, refresh), AugmentationPlan(11, refresh, refresh)
            first.generators("b", 0)
            self.assertEqual(first.generators("a", 0)[0].random(), second.generators("a", 0)[0].random())

    def test_transform_settings_never_change_native_crop_position(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            value = np.arange(64 * 64, dtype=np.uint16).reshape(64, 64) % 256
            Image.fromarray(np.repeat(value[..., None].astype(np.uint8), 3, axis=2)).save(root / "diff.png")
            Image.fromarray((value * 257).astype(np.uint16)).save(root / "disp.png")
            sample = {"id": "pair", "input": str(root / "diff.png"), "target": str(root / "disp.png")}
            records = []
            for flip, rotate in ((False, False), (True, False), (True, True)):
                crop, transform = AugmentationPlan(4, "per_epoch", "per_epoch").generators("pair", 0)
                rgb, target, record = native_pair(sample, 32, crop, "depth", flip, rotate, False,
                                                  augmentation_rng=transform)
                torch.testing.assert_close(rgb[0], target[0])
                records.append((record["x"], record["y"]))
            self.assertEqual(len(set(records)), 1)
            _, _, centered = native_pair(sample, 32, random.Random(8), "depth", False, False, False, center_crop=True)
            self.assertEqual((centered["x"], centered["y"]), (16, 16))

    def test_per_run_crops_and_flips_stay_fixed_when_epoch_order_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            value = np.arange(64 * 64, dtype=np.uint16).reshape(64, 64) % 256
            Image.fromarray(np.repeat(value[..., None].astype(np.uint8), 3, axis=2)).save(root / "diff.png")
            Image.fromarray((value * 257).astype(np.uint16)).save(root / "disp.png")
            plan = AugmentationPlan(42, "per_run", "per_run")
            epochs = []
            for epoch, order in enumerate((("a", "b", "c"), ("c", "a", "b"))):
                records = {}
                for sample_id in order:
                    sample = {"id": sample_id, "input": str(root / "diff.png"), "target": str(root / "disp.png")}
                    crop, transform = plan.generators(sample_id, epoch)
                    _, _, record = native_pair(sample, 32, crop, "depth", True, True, False, augmentation_rng=transform)
                    records[sample_id] = record
                epochs.append(records)
            self.assertEqual(epochs[0], epochs[1])


class TrainingControlTests(unittest.TestCase):
    def fixture(self, root, count=5, validation=False):
        samples = []
        for index in range(count):
            material = root / f"material-{index}"
            material.mkdir()
            rgb = np.random.default_rng(index).integers(0, 256, (32, 32, 3), dtype=np.uint8)
            Image.fromarray(rgb).save(material / "diff.png")
            Image.fromarray(rgb[..., 0].astype(np.uint16) * 257).save(material / "disp.png")
            samples.append({"id": str(index), "input": str(material / "diff.png"),
                            "target": str(material / "disp.png"),
                            "split": "validation" if validation and index == count - 1 else "training"})
        return {"schema": "loom-factory-training-v1", "project_id": "project", "dataset_id": "bark",
                "source_root": str(root), "map_kind": "depth", "image_size": 32, "width": 8, "blocks": 1,
                "epochs": 1, "batch_size": 2, "shuffle_each_epoch": False, "flip_crops": False,
                "rotate_crops": False, "device": "cpu", "optimizer": "sgd", "momentum": 0,
                "weight_decay": 0, "gradient_clip_norm": 0, "seed": 8, "learning_rate": .02, "samples": samples}

    def run_fixture(self, request, root, run):
        request = {**request, "output_directory": str(root / ".loom-factory" / "runs" / run)}
        with patch("loom_factory.engine.export_coreml", fake_export), contextlib.redirect_stdout(io.StringIO()) as log:
            descriptor = train(request)
        return descriptor, [json.loads(line) for line in log.getvalue().splitlines()], Path(request["output_directory"])

    def test_sample_weighted_accumulation_matches_whole_batch_and_partial_windows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = self.fixture(root)
            reference, _, reference_path = self.run_fixture({**request, "batch_size": 4}, root, "whole")
            accumulated, events, accumulated_path = self.run_fixture({**request, "gradient_accumulation_steps": 2}, root, "accumulated")
            for key, value in load_file(str(reference_path / "model.safetensors")).items():
                torch.testing.assert_close(value, load_file(str(accumulated_path / "model.safetensors"))[key], rtol=2e-5, atol=2e-7)
            progress = [event for event in events if event["event"] == "progress"]
            self.assertEqual([event["update"] for event in progress], [0, 1, 2])
            self.assertEqual([event["optimizer_step"] for event in progress], [False, True, True])
            self.assertEqual(accumulated["training"]["updates"], 2)
            self.assertEqual(accumulated["training"]["effective_batch_size"], 4)
            self.assertEqual(progress[-1]["progress"], 1)
            self.assertEqual(accumulated["parameter_count"], sum(p.numel() for p in MaterialMapNetwork("depth", 8, 1).parameters()))

    def test_uneven_final_microbatch_matches_one_mean_gradient(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = self.fixture(root)
            _, _, whole = self.run_fixture({**request, "batch_size": 5}, root, "whole")
            descriptor, _, accumulated = self.run_fixture({**request, "gradient_accumulation_steps": 3}, root, "accumulated")
            self.assertEqual(descriptor["training"]["updates"], 1)
            for key, value in load_file(str(whole / "model.safetensors")).items():
                torch.testing.assert_close(value, load_file(str(accumulated / "model.safetensors"))[key], rtol=2e-5, atol=2e-7)

    def test_validation_best_export_early_stopping_and_checkpoint_interval(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = {**self.fixture(root, validation=True), "epochs": 6, "checkpoint_every_epochs": 4,
                       "early_stopping_patience": 2, "export_best_weights": True}
            with patch("loom_factory.engine.validate_model", side_effect=[.2, .3, .4]):
                descriptor, events, output = self.run_fixture(request, root, "validated")
            self.assertEqual(descriptor["training"]["epochs"], 3)
            self.assertEqual(descriptor["training"]["requested_epochs"], 6)
            self.assertTrue(descriptor["training"]["stopped_early"])
            self.assertEqual(descriptor["training"]["export_epoch"], 1)
            self.assertEqual(descriptor["final_validation_loss"], .2)
            checkpoint = json.loads((output / "checkpoint.json").read_text())
            self.assertEqual(checkpoint["epochs_completed"], 3)
            self.assertEqual(checkpoint["settings"]["gradient_accumulation_steps"], 1)
            self.assertEqual(checkpoint["status"], "completed")
            for key, value in load_file(str(output / "best.safetensors")).items():
                torch.testing.assert_close(value, load_file(str(output / "model.safetensors"))[key])
            self.assertNotEqual((output / "checkpoint.safetensors").read_bytes(), (output / "best.safetensors").read_bytes())
            self.assertEqual(len([event for event in events if event["event"] == "epoch_completed"]), 3)

    def test_cancel_during_best_conversion_preserves_latest_continuation_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = {**self.fixture(root, validation=True), "epochs": 2,
                       "output_directory": str(root / ".loom-factory" / "runs" / "cancel-best")}
            cancellation = Cancellation()
            def cancel_export(model, output, size):
                fake_export(model, output, size)
                cancellation.requested = True
            with patch("loom_factory.engine.validate_model", side_effect=[.2, .3]), \
                    patch("loom_factory.engine.export_coreml", cancel_export), \
                    contextlib.redirect_stdout(io.StringIO()), self.assertRaises(TrainingCancelled):
                train(request, cancellation)
            output = Path(request["output_directory"])
            checkpoint = json.loads((output / "checkpoint.json").read_text())
            self.assertEqual(checkpoint["status"], "cancelled")
            self.assertEqual(checkpoint["epochs_completed"], 2)
            self.assertNotEqual((output / "checkpoint.safetensors").read_bytes(), (output / "best.safetensors").read_bytes())
            self.assertFalse((output / "model.json").exists())

    def test_real_validation_is_fixed_held_out_and_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            descriptor, events, _ = self.run_fixture(self.fixture(root, validation=True), root, "real-validation")
            self.assertEqual(descriptor["training"]["training_sample_count"], 4)
            self.assertEqual(descriptor["training"]["validation_sample_count"], 1)
            self.assertTrue(np.isfinite(descriptor["final_validation_loss"]))
            self.assertEqual(descriptor["training"]["validation_crop"], "fixed-center-native-no-augmentation")
            seen = [sample for event in events if event["event"] == "progress" for sample in event["sample_ids"]]
            self.assertNotIn("4", seen)

    @unittest.skipUnless(torch.backends.mps.is_available(), "Apple GPU is unavailable")
    def test_each_optimizer_runs_accumulation_and_validation_on_apple_gpu(self):
        for optimizer in ("adamw", "adam", "sgd", "lion"):
            with self.subTest(optimizer=optimizer), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                request = {**self.fixture(root, validation=True), "device": "mps", "optimizer": optimizer,
                           "epochs": 2, "batch_size": 1, "gradient_accumulation_steps": 3,
                           "lr_schedule": "cosine", "learning_rate": .0001}
                descriptor, events, output = self.run_fixture(request, root, "apple-gpu")
                self.assertEqual(descriptor["training"]["updates"], 4)
                self.assertEqual(descriptor["training"]["device"], "mps")
                self.assertTrue(np.isfinite(descriptor["final_validation_loss"]))
                self.assertTrue(all(torch.isfinite(value).all() for value in load_file(str(output / "model.safetensors")).values()))
                progress = [event for event in events if event["event"] == "progress"]
                self.assertAlmostEqual(progress[-1]["learning_rate"], .00001)

    def test_validation_rejects_resolution_leakage_and_early_stop_without_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = self.fixture(root, count=2)
            request["output_directory"] = str(root / ".loom-factory" / "runs" / "invalid")
            original = request["samples"][0]
            material = Path(original["input"]).parent
            for name, original_field in (("second-diff.png", "input"), ("second-disp.png", "target")):
                (material / name).write_bytes(Path(original[original_field]).read_bytes())
            leaked = {**original, "id": "different-resolution", "input": str(material / "second-diff.png"),
                      "target": str(material / "second-disp.png"), "split": "validation"}
            with self.assertRaisesRegex(ValueError, "same training or validation split"):
                validate_request({**request, "samples": [*request["samples"], leaked]})
            with self.assertRaisesRegex(ValueError, "requires held-out validation"):
                validate_request({**request, "early_stopping_patience": 1})
            alias_folder = root / "alias-material"
            alias_folder.mkdir()
            for field in ("input", "target"):
                (alias_folder / Path(original[field]).name).symlink_to(original[field])
            alias = {**original, "id": "alias", "input": str(alias_folder / "diff.png"),
                     "target": str(alias_folder / "disp.png"), "split": "validation"}
            with self.assertRaisesRegex(ValueError, "including through symbolic links"):
                validate_request({**request, "samples": [*request["samples"], alias]})

    def test_unmarked_cli_samples_remain_all_training_without_automatic_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = self.fixture(root, count=2)
            for sample in request["samples"]:
                sample.pop("split")
            descriptor, events, output = self.run_fixture(request, root, "legacy-cli")
            self.assertEqual(descriptor["training"]["training_sample_count"], 2)
            self.assertEqual(descriptor["training"]["validation_sample_count"], 0)
            self.assertIsNone(descriptor["final_validation_loss"])
            self.assertFalse((output / "best.safetensors").exists())
            self.assertIsNone([event for event in events if event["event"] == "epoch_completed"][0]["validation_loss"])

    def tag_fixture(self, request):
        request["dataset_id"] = "dataset-" + hashlib.sha256(b"bark").hexdigest()[:16]
        request["require_image_metadata"] = True
        tags = {}
        for sample in request["samples"]:
            for field, role in (("input", "diffuse"), ("target", "depth")):
                tag = {"schema": "loom-factory.image.v1", "id": sample["id"] + field,
                       "category": "Bark", "role": role, "pairingKey": sample["id"],
                       "verified": True, "ignored": False, "note": "Original review"}
                write_fixture_metadata(sample[field], tag)
                tags[sample[field]] = tag
        return tags

    def test_changed_source_pixels_stop_before_the_next_optimizer_update(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = {**self.fixture(root, count=1), "epochs": 2, "batch_size": 1,
                       "output_directory": str(root / ".loom-factory" / "runs" / "changed-pixels")}
            tags = self.tag_fixture(request)
            events = []
            def mutate_after_first_update(event, **fields):
                events.append({"event": event, **fields})
                if event == "progress":
                    path = request["samples"][0]["input"]
                    Image.fromarray(np.full((32, 32, 3), 255, dtype=np.uint8)).save(path)
                    write_fixture_metadata(path, tags[path])
            with patch("loom_factory.engine.emit", mutate_after_first_update), \
                    patch("loom_factory.engine.export_coreml", fake_export), \
                    self.assertRaisesRegex(ValueError, "source image changed during training"):
                train(request)
            progress = [event for event in events if event["event"] == "progress"]
            self.assertEqual([event["update"] for event in progress], [1])
            output = Path(request["output_directory"])
            checkpoint = json.loads((output / "checkpoint.json").read_text())
            self.assertEqual(checkpoint["updates"], 1)
            self.assertEqual(checkpoint["status"], "stopped")
            self.assertFalse((output / "model.json").exists())

    def test_review_note_metadata_changes_do_not_look_like_pixel_modification(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = {**self.fixture(root, count=1), "epochs": 2, "batch_size": 1,
                       "output_directory": str(root / ".loom-factory" / "runs" / "note-only")}
            tags = self.tag_fixture(request)
            original_times = {path: Path(path).stat().st_mtime_ns for path in tags}
            def change_note(event, **fields):
                if event == "progress" and fields["update"] == 1:
                    for path, tag in tags.items():
                        write_fixture_metadata(path, {**tag, "note": "Updated review note"})
            with patch("loom_factory.engine.emit", change_note), patch("loom_factory.engine.export_coreml", fake_export):
                descriptor = train(request)
            self.assertEqual(descriptor["training"]["updates"], 2)
            self.assertEqual(original_times, {path: Path(path).stat().st_mtime_ns for path in tags})

    def test_unreviewed_pairs_train_and_review_flag_changes_preserve_membership(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = {**self.fixture(root, count=1), "epochs": 3, "batch_size": 1,
                       "output_directory": str(root / ".loom-factory" / "runs" / "review-informational")}
            tags = self.tag_fixture(request)
            for path, tag in tags.items():
                write_fixture_metadata(path, {**tag, "verified": False})
            updates = []
            def toggle_review(event, **fields):
                if event == "progress":
                    updates.append(fields["update"])
                    for path, tag in tags.items():
                        write_fixture_metadata(path, {**tag, "verified": fields["update"] == 1})
            with patch("loom_factory.engine.emit", toggle_review), patch("loom_factory.engine.export_coreml", fake_export):
                descriptor = train(request)
            self.assertEqual(updates, [1, 2, 3])
            self.assertEqual(descriptor["training"]["updates"], 3)
            self.assertTrue((Path(request["output_directory"]) / "model.json").is_file())

    def test_changed_validation_pixels_stop_before_held_out_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = {**self.fixture(root, count=2, validation=True), "batch_size": 1,
                       "output_directory": str(root / ".loom-factory" / "runs" / "changed-validation")}
            tags = self.tag_fixture(request)
            events = []
            def change_validation_pixels(event, **fields):
                events.append({"event": event, **fields})
                if event == "progress":
                    path = request["samples"][-1]["target"]
                    Image.fromarray(np.zeros((32, 32), dtype=np.uint16)).save(path)
                    write_fixture_metadata(path, tags[path])
            with patch("loom_factory.engine.emit", change_validation_pixels), \
                    patch("loom_factory.engine.export_coreml", fake_export), \
                    self.assertRaisesRegex(ValueError, "source image changed during training"):
                train(request)
            self.assertFalse(any(event["event"] == "epoch_completed" for event in events))
            self.assertFalse((Path(request["output_directory"]) / "best.safetensors").exists())


if __name__ == "__main__":
    unittest.main()
