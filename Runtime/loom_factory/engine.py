"""Training/export orchestration. Requests and artifacts live with their source project."""

from contextlib import redirect_stdout
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import time

import torch
from torch.nn import functional as F
from safetensors.torch import load_file, save_file

from .images import native_pair
from .metadata import read_image_metadata
from .models import MaterialMapNetwork
from .pixel_cache import DecodedImageCache


class TrainingCancelled(Exception):
    pass


class Cancellation:
    def __init__(self):
        self.requested = False

    def signal(self, *_):
        self.requested = True

    def check(self):
        if self.requested:
            raise TrainingCancelled()


def emit(event: str, **fields):
    print(json.dumps({"event": event, **fields}, allow_nan=False), flush=True)


def write_json(path: Path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def file_sha(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def integer(request, name, default, low, high):
    value = request.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer from {low} to {high}.")
    return value


def validate_request(request: dict):
    if request.get("schema") != "loom-factory-training-v1":
        raise ValueError("This training request uses an unsupported schema.")
    source = Path(os.path.abspath(os.path.expanduser(request["source_root"])))
    if not source.is_dir():
        raise ValueError("The source folder no longer exists.")
    output = Path(os.path.abspath(os.path.expanduser(request["output_directory"])))
    project_storage = source.resolve() / ".loom-factory"
    if not output.resolve().is_relative_to(project_storage):
        raise ValueError("Training output must be inside this source folder's .loom-factory directory.")
    if output.resolve() == project_storage:
        raise ValueError("Each training run needs its own output directory.")
    for key in ("project_id", "dataset_id"):
        if not request.get(key):
            raise ValueError(f"Training requires {key}.")
    samples = request.get("samples", [])
    if not samples:
        raise ValueError("This dataset has no images for the requested training type.")
    ids = set()
    paths = {}
    for sample_index, sample in enumerate(samples):
        sample_id = str(sample["id"])
        if sample_id in ids:
            raise ValueError(f"The training request repeats an image pair: {sample_id}")
        ids.add(sample_id)
        for field in ("input", "target"):
            path = Path(os.path.abspath(os.path.expanduser(sample[field])))
            # Symlinked originals are allowed, but the link itself must live in
            # the selected source folder. A cached external path is not a member.
            if not path.is_relative_to(source) or path.is_relative_to(source / ".loom-factory"):
                raise ValueError(f"Image path is outside the current source folder: {path.name}")
            if not path.is_file():
                raise ValueError(f"An image was moved or deleted after the folder scan: {path.name}")
            sample[field] = str(path)
            stat = path.stat()
            paths[str(path)] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                                "sha256": file_sha(path)}
        if sample_index % 20 == 0 or sample_index + 1 == len(samples):
            emit("status", message=f"Checking source images {sample_index + 1} of {len(samples)}…")
    size = integer(request, "image_size", 1024, 32, 8192)
    if request.get("map_kind") not in ("depth", "roughness", "normal"):
        raise ValueError("Map training must specify depth, roughness or normal.")
    if request.get("exaggerate_features", False) and request.get("map_kind") != "depth":
        raise ValueError("Exaggerate features is available for depth training.")
    integer(request, "epochs", 10, 1, 100000)
    integer(request, "batch_size", 1, 1, 256)
    integer(request, "image_cache_megabytes", 256, 0, 2048)
    if "width" not in request and "model_width" in request:
        request["width"] = request["model_width"]
    integer(request, "width", 24, 8, 256)
    integer(request, "blocks", 4, 1, 12)
    rate = request.get("learning_rate", 0.001)
    if isinstance(rate, bool) or not isinstance(rate, (float, int)) or not math.isfinite(rate) or not 0 < rate <= 1:
        raise ValueError("The learning rate must be greater than zero and at most one.")
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ("model.json", "model.safetensors", "model.mlpackage")):
        raise ValueError("This run already contains a model. Start a new run to preserve it.")
    return source, output, samples, paths, size


def current_image_tags(request, sample):
    """Stop a run when its current file metadata no longer authorizes a sample."""
    if request["map_kind"] == "normal" and sample.get("normal_convention", "OpenGL").lower() == "directx":
        raise ValueError("DirectX normal maps are excluded from training.")
    for field in ("input", "target"):
        path = Path(sample[field])
        if not path.is_file():
            raise ValueError(f"An image was moved or deleted during training: {path.name}")
        metadata = read_image_metadata(path)
        if metadata is None:
            # Untagged CLI fixtures are permitted explicitly. The app sets
            # require_image_metadata, so a removed tag cannot become an opt-out.
            if not request.get("require_image_metadata", False):
                continue
            raise ValueError(f"The image's review metadata changed during training: {path.name}")
        category = str(metadata.get("category") or "").strip()
        category_id = "dataset-" + hashlib.sha256(category.lower().encode()).hexdigest()[:16]
        role = "diffuse" if field == "input" else request["map_kind"]
        if role == "normal" and (metadata.get("normalConvention") or "opengl").lower() == "directx":
            raise ValueError(f"DirectX normal maps are excluded from training: {path.name}")
        if (metadata.get("schema") != "loom-factory.image.v1" or not metadata.get("verified") or
                metadata.get("ignored", False) or metadata.get("role") != role or category_id != request["dataset_id"] or
                metadata.get("pairingKey") != sample["id"]):
            raise ValueError(f"The dataset tag, map type or review state changed during training: {path.name}")


def unchanged_sources(paths):
    for path, expected in paths.items():
        actual = Path(path)
        if not actual.is_file() or file_sha(actual) != expected["sha256"]:
            raise ValueError(f"A source image changed during training: {actual.name}. Start a run from the refreshed folder.")


def device_for(name):
    if name == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if name == "mps" and not torch.backends.mps.is_available():
        raise ValueError("Apple GPU training is unavailable on this Mac.")
    if name not in ("mps", "cpu"):
        raise ValueError("Training device must be auto, mps or cpu.")
    return torch.device(name)


def map_loss(prediction, target, kind):
    if kind == "normal":
        prediction_vector = F.normalize(prediction * 2 - 1, dim=1, eps=1e-6)
        target_vector = F.normalize(target * 2 - 1, dim=1, eps=1e-6)
        return (1 - (prediction_vector * target_vector).sum(dim=1)).mean()
    return F.l1_loss(prediction, target)


def export_coreml(model, path, image_size):
    # All conversion diagnostics go to stderr; stdout is a JSON-lines protocol.
    with redirect_stdout(sys.stderr):
        import coremltools as ct
        model = model.cpu().eval()
        example = torch.zeros(1, 3, 64, 64)
        traced = torch.jit.trace(model, example)
        shape = (1, 3, ct.RangeDim(32, 8192, default=image_size), ct.RangeDim(32, 8192, default=image_size))
        output = ct.TensorType(name="material_map")
        converted = ct.convert(
            traced, inputs=[ct.TensorType(name="rgb", shape=shape)], outputs=[output],
            convert_to="mlprogram", compute_precision=ct.precision.FLOAT32,
            minimum_deployment_target=ct.target.macOS13,
        )
        converted.short_description = "Texture Loom model trained in Loom Factory"
        converted.input_description["rgb"] = "Stored diffuse RGB sRGB pixels, float32 NCHW, normalized once to [0,1]."
        converted.user_defined_metadata["loom.schema"] = "loom-factory-model-v1"
        converted.user_defined_metadata["loom.architecture"] = model.architecture
        converted.save(str(path))
    return converted


def artifact_hashes(output):
    files = [output / "model.safetensors"] + sorted((output / "model.mlpackage").rglob("*"))
    return [{"path": str(path.relative_to(output)), "sha256": file_sha(path)}
            for path in files if path.is_file()]


def save_checkpoint(model, request, output, history, updates, status):
    if not updates:
        return
    weights = {key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}
    if not all(torch.isfinite(value).all() for value in weights.values()):
        return  # Preserve the last finite checkpoint if a device operation failed.
    temporary = output / "checkpoint.safetensors.tmp"
    save_file(weights, str(temporary), metadata={"schema": "loom-factory-checkpoint-v1", "architecture": model.architecture})
    temporary.replace(output / "checkpoint.safetensors")
    descriptor = {"schema": "loom-factory-checkpoint-v1", "status": status,
                  "project_id": request.get("project_id"), "dataset_id": request.get("dataset_id"),
                  "dataset_name": request.get("dataset_name", request.get("display_name")),
                  "kind": "map", "map": request["map_kind"],
                  "architecture": model.architecture, "architecture_config": model.descriptor(),
                  "settings": {key: request.get(key, default) for key, default in (
                      ("image_size", 1024), ("width", 24),
                      ("blocks", 4), ("exaggerate_features", False), ("shuffle_each_epoch", True),
                      ("flip_crops", True), ("rotate_crops", False))},
                  "epochs_completed": len(history), "updates": updates, "history": history,
                  "weights_path": "checkpoint.safetensors", "weights_sha256": file_sha(output / "checkpoint.safetensors"),
                  "optimizer": "fresh optimizer on continuation",
                  "created_at": datetime.now(timezone.utc).isoformat()}
    write_json(output / "checkpoint.json", descriptor)
    emit("checkpoint", message="Training weights saved in this source project.",
         weights_path=str(output / "checkpoint.safetensors"), updates=updates, epoch=len(history))


def continue_weights(model, request, source):
    value = request.get("resume_weights")
    if not value:
        return
    weights = Path(os.path.abspath(os.path.expanduser(value)))
    if not weights.resolve().is_relative_to(source.resolve() / ".loom-factory" / "runs"):
        raise ValueError("Continued weights must belong to this source project's training runs.")
    descriptor_path = weights.parent / "checkpoint.json"
    if weights.name != "checkpoint.safetensors" or not weights.is_file() or not descriptor_path.is_file():
        raise ValueError("Choose saved training weights with their checkpoint descriptor.")
    checkpoint = json.loads(descriptor_path.read_text())
    expected = {"project_id": request.get("project_id"), "dataset_id": request.get("dataset_id"),
                "kind": "map", "map": request["map_kind"]}
    if checkpoint.get("schema") != "loom-factory-checkpoint-v1" or any(checkpoint.get(key) != value for key, value in expected.items()):
        raise ValueError("These saved weights belong to a different project, dataset or map type.")
    if checkpoint.get("architecture_config") != model.descriptor():
        raise ValueError("Continue with the same network width and block count as the saved weights.")
    if checkpoint.get("settings", {}).get("image_size") != request.get("image_size", 1024):
        raise ValueError("Continue with the same native crop size as the saved weights.")
    if bool(checkpoint.get("settings", {}).get("exaggerate_features")) != bool(request.get("exaggerate_features", False)):
        raise ValueError("Continue with the same Exaggerate features setting as the saved weights.")
    if checkpoint.get("weights_sha256") != file_sha(weights):
        raise ValueError("The saved checkpoint was not written completely. Use another run's saved weights.")
    loaded = load_file(str(weights))
    if not all(torch.isfinite(value).all() for value in loaded.values()):
        raise ValueError("The saved checkpoint contains non-finite weights.")
    model.load_state_dict(loaded, strict=True)
    emit("status", message="Continuing saved weights with a fresh optimizer and current dataset images.")


def train(request, cancellation=None):
    cancellation = cancellation or Cancellation()
    start = time.monotonic()
    emit("status", message="Checking the current source files…")
    source, output, samples, fingerprints, image_size = validate_request(request)
    for sample in samples:
        current_image_tags(request, sample)
    cancellation.check()
    seed = request.get("seed", 42)
    torch.manual_seed(seed)
    device = device_for(request.get("device", "auto"))
    if device.type == "mps":
        torch.mps.manual_seed(seed)
    if device.type == "cpu":
        torch.set_num_threads(min(8, os.cpu_count() or 1))
    width = request.get("width", 24)
    model = MaterialMapNetwork(request["map_kind"], width, request.get("blocks", 4))
    continue_weights(model, request, source)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=request.get("learning_rate", 0.001))
    epochs, batch_size = request.get("epochs", 10), request.get("batch_size", 1)
    shuffle = request.get("shuffle_each_epoch", True)
    order_rng, crop_rng = random.Random(seed), random.Random(seed + 1)
    image_cache = DecodedImageCache(request.get("image_cache_megabytes", 256) * 1024 * 1024)
    history, update, total_updates = [], 0, epochs * math.ceil(len(samples) / batch_size)
    emit("started", device=device.type, sample_count=len(samples), image_size=image_size,
         epochs=epochs, total_updates=total_updates, architecture=model.descriptor())
    try:
        for epoch in range(epochs):
            cancellation.check()
            order = list(range(len(samples)))
            if shuffle:
                order_rng.shuffle(order)
            epoch_loss, epoch_count = 0.0, 0
            for offset in range(0, len(order), batch_size):
                cancellation.check()
                batch = [samples[index] for index in order[offset:offset + batch_size]]
                inputs, targets, crop_records = [], [], []
                for sample in batch:
                    current_image_tags(request, sample)
                    rgb, target, crop = native_pair(
                        sample, image_size, crop_rng, request["map_kind"],
                        request.get("flip_crops", True), request.get("rotate_crops", False),
                        request.get("exaggerate_features", False),
                        cache=image_cache,
                    )
                    inputs.append(rgb)
                    targets.append(target)
                    crop_records.append({"id": sample["id"], **crop})
                cancellation.check()
                input_batch = torch.stack(inputs).to(device)
                target_batch = torch.stack(targets).to(device)
                optimizer.zero_grad(set_to_none=True)
                prediction = model(input_batch)
                loss = map_loss(prediction, target_batch, request["map_kind"])
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                # One small device transfer checks both values and supplies real
                # progress, instead of synchronizing the GPU three times.
                loss_number, gradient_number = torch.stack((loss.detach(), gradient_norm.detach())).cpu().tolist()
                if not math.isfinite(loss_number):
                    raise ValueError("Training produced a non-finite loss. The run was stopped before exporting a model.")
                if not math.isfinite(gradient_number):
                    raise ValueError("Training produced non-finite gradients. The last finite checkpoint was preserved.")
                optimizer.step()
                update += 1
                epoch_loss += loss_number * len(batch)
                epoch_count += len(batch)
                emit("progress", epoch=epoch + 1, epochs=epochs, update=update,
                     total_updates=total_updates, loss=loss_number, sample_ids=[sample["id"] for sample in batch],
                     crops=crop_records, progress=update / total_updates,
                     message=f"Epoch {epoch + 1} of {epochs} · loss {loss_number:.5f}")
                del prediction, loss, input_batch, target_batch, inputs, targets
            order_ids = [str(samples[index]["id"]) for index in order]
            epoch_result = {"epoch": epoch + 1, "loss": epoch_loss / epoch_count,
                            "order_sha256": hashlib.sha256(json.dumps(order_ids).encode()).hexdigest(),
                            "first_sample_ids": order_ids[:20]}
            history.append(epoch_result)
            save_checkpoint(model, request, output, history, update, "training")
            write_json(output / "progress.json", {"status": "training", "history": history, "updates": update})
        cancellation.check()
        for sample in samples:
            current_image_tags(request, sample)
        unchanged_sources(fingerprints)
        emit("status", message="Exporting the model for Texture Studio…")
        weights = {key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}
        temporary_weights = output / "model.safetensors.tmp"
        save_file(weights, str(temporary_weights), metadata={"architecture": model.architecture,
                                                          "schema": "loom-factory-model-v1"})
        temporary_weights.replace(output / "model.safetensors")
        cancellation.check()
        export_coreml(model, output / "model.mlpackage", image_size)
        cancellation.check()
        for sample in samples:
            current_image_tags(request, sample)
        unchanged_sources(fingerprints)
        descriptor = {
            "schema": "loom-factory-model-v1", "status": "completed", "kind": "map",
            "project_id": request.get("project_id"), "dataset_id": request.get("dataset_id"),
            "dataset_name": request.get("dataset_name", request.get("display_name")),
            "map": request["map_kind"], "architecture": model.architecture,
            "architecture_config": model.descriptor(), "final_loss": history[-1]["loss"],
            "image_size": image_size, "input_name": "rgb", "output_name": "material_map",
            "coreml_path": "model.mlpackage", "weights_path": "model.safetensors",
            "training": {"epochs": epochs, "updates": update, "batch_size": batch_size,
                         "initialization": "continued-project-weights" if request.get("resume_weights") else "random",
                         "shuffle_each_epoch": shuffle, "exaggerate_features": request.get("exaggerate_features", False),
                         "depth_transform": "anchored-tanh-contrast-strength-2" if request.get("exaggerate_features", False) else "identity",
                         "seed": seed, "device": device.type, "history": history,
                         "decoded_image_cache": image_cache.statistics()},
            "sources": [{"path": str(Path(path).relative_to(source)), **proof} for path, proof in fingerprints.items()],
            "files": artifact_hashes(output), "created_at": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": time.monotonic() - start,
            "quality_note": "Training completion is not a validation of prediction quality on unseen materials.",
        }
        checkpoint = json.loads((output / "checkpoint.json").read_text())
        checkpoint["status"] = "completed"
        write_json(output / "checkpoint.json", checkpoint)
        write_json(output / "progress.json", {"status": "completed", "history": history, "updates": update})
        # Publish the deployment descriptor last. Only this file authorizes
        # Factory to register a completed model for package assembly.
        write_json(output / "model.json", descriptor)
        emit("completed", model_path=str(output / "model.json"), coreml_path=str(output / "model.mlpackage"),
             weights_path=str(output / "model.safetensors"), updates=update, duration_seconds=time.monotonic() - start)
        return descriptor
    except TrainingCancelled:
        try:
            save_checkpoint(model, request, output, history, update, "cancelled")
        except Exception as checkpoint_error:
            emit("warning", message=f"Latest checkpoint weights could not be saved: {checkpoint_error}")
        write_json(output / "progress.json", {"status": "cancelled", "history": history, "updates": update})
        emit("cancelled", updates=update, message="Training stopped. Source images were preserved.")
        raise
    except Exception as error:
        try:
            save_checkpoint(model, request, output, history, update, "stopped")
        except Exception as checkpoint_error:
            emit("warning", message=f"Latest checkpoint weights could not be saved: {checkpoint_error}")
        write_json(output / "progress.json", {"status": "failed", "history": history, "updates": update,
                                               "message": str(error)})
        raise


def run_cli(command: str, request_path: str):
    cancellation = Cancellation()
    for kind in (signal.SIGTERM, signal.SIGINT):
        signal.signal(kind, cancellation.signal)
    try:
        if command != "train-map":
            raise ValueError("This runtime trains material maps only. Train image classifiers in Create ML.")
        request = json.loads(Path(request_path).read_text())
        train(request, cancellation=cancellation)
        return 0
    except TrainingCancelled:
        return 130
    except Exception as error:
        emit("error", message=str(error), error_type=type(error).__name__)
        return 1
