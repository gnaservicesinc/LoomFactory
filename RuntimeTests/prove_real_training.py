"""Opt-in real 2048 training and native deployment proof on copied source pixels."""

import argparse
import contextlib
import hashlib
import json
from pathlib import Path
import shutil
import sys
import uuid

import coremltools as ct
import numpy as np
from PIL import Image
import torch
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Runtime"))
from loom_factory.engine import file_sha, train
from loom_factory.images import read_rgb
from loom_factory.metadata import write_fixture_metadata
from loom_factory.models import MaterialMapNetwork


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-folder", type=Path, required=True)
    parser.add_argument("--validation-folder", type=Path, required=True)
    args = parser.parse_args()
    original = args.source_folder / "brick_wall_005"
    input_original = original / "brick_wall_005_diff_2k.png"
    target_original = original / "brick_wall_005_disp_2k.png"
    proof_root = args.validation_folder
    source = proof_root / "Source"
    fixture = source / "Bricks"
    fixture.mkdir(parents=True, exist_ok=True)
    category_id = "dataset-" + hashlib.sha256(b"bricks").hexdigest()[:16]
    original_hashes = {str(path): file_sha(path) for path in (input_original, target_original)}
    input_path, target_path = fixture / input_original.name, fixture / target_original.name
    for actual, copy, role in ((input_original, input_path, "diffuse"), (target_original, target_path, "depth")):
        shutil.copy2(actual, copy)
        write_fixture_metadata(copy, {"schema": "loom-factory.image.v1", "id": str(uuid.uuid4()),
                                     "pairingKey": "real-brick-pair", "role": role,
                                     "category": "Bricks", "verified": True, "note": "Real native2048 validation copy"})
    with Image.open(target_path) as image:
        target_mode, target_size = image.mode, image.size
    run = source / ".loom-factory" / "runs" / ("real-two-k-" + uuid.uuid4().hex)
    request = {"schema": "loom-factory-training-v1", "project_id": "fresh-runtime-validation",
               "dataset_id": category_id, "dataset_name": "Bricks", "source_root": str(source),
               "output_directory": str(run), "map_kind": "depth", "image_size": 2048,
               "epochs": 1, "batch_size": 1, "model_width": 8, "blocks": 1,
               "shuffle_each_epoch": False, "flip_crops": False, "rotate_crops": False,
               "exaggerate_features": True, "require_image_metadata": True,
               "device": "auto", "samples": [{"id": "real-brick-pair", "input": str(input_path), "target": str(target_path)}]}
    run.mkdir(parents=True, exist_ok=True)
    (run / "request.json").write_text(json.dumps(request, indent=2) + "\n")
    with (run / "events.jsonl").open("w") as log, contextlib.redirect_stdout(log):
        result = train(request)
    deployed = ct.models.MLModel(str(run / "model.mlpackage"), compute_units=ct.ComputeUnit.CPU_ONLY)
    rgb = read_rgb(str(input_path))[:64, :64].transpose(2, 0, 1)[None].copy()
    predicted = deployed.predict({"rgb": rgb})["material_map"]
    reference = MaterialMapNetwork("depth", 8, 1).eval()
    reference.load_state_dict(load_file(str(run / "model.safetensors")))
    expected = reference(torch.from_numpy(rgb)).detach().numpy()
    maximum_difference = float(np.max(np.abs(predicted - expected)))
    np.testing.assert_allclose(predicted, expected, atol=3e-4, rtol=3e-4)
    if not np.isfinite(predicted).all():
        raise AssertionError("Native prediction must contain finite output pixels.")
    preserved = all(file_sha(Path(path)) == sha for path, sha in original_hashes.items())
    if not preserved:
        raise AssertionError("An original source image changed.")
    proof = {"schema": "loom-factory-real-training-proof-v1", "run": str(run),
             "image_size": 2048, "target_file_mode": target_mode, "target_native_size": target_size,
             "device": result["training"]["device"], "updates": result["training"]["updates"],
             "fixed_order": not result["training"]["shuffle_each_epoch"],
             "depth_transform": result["training"]["depth_transform"],
             "original_hashes": original_hashes, "originals_preserved": preserved,
             "coreml_predicted_shape": list(predicted.shape), "coreml_maximum_absolute_difference": maximum_difference,
             "quality_qualification": False,
             "qualification_note": "One real update proves native2048 execution and export, not trained model quality."}
    (proof_root / "real-two-k-training-proof.json").write_text(json.dumps(proof, indent=2) + "\n")
    print(json.dumps(proof, indent=2))


if __name__ == "__main__":
    main()
