"""Compare repeated native paired-crop loading on disposable 2K sources.

This isolates pixel preparation; it does not measure end-to-end model training.
Run with the private training Python. No user images or project state are read.
"""
import json
from pathlib import Path
import random
import sys
import tempfile
import time

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Runtime"))
from loom_factory.images import native_pair
from loom_factory.pixel_cache import DecodedImageCache


def benchmark():
    torch.set_num_threads(4)
    iterations = 12
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        rng = np.random.default_rng(42)
        Image.fromarray(rng.integers(0, 256, (2048, 2048, 3), dtype=np.uint8)).save(root / "diffuse.png")
        Image.fromarray(rng.integers(0, 65536, (2048, 2048), dtype=np.uint16)).save(root / "depth.png")
        sample = {"id": "disposable", "input": str(root / "diffuse.png"), "target": str(root / "depth.png")}
        # Warm filesystem/imports, retaining no decoded cache for the baseline.
        native_pair(sample, 1024, random.Random(3), "depth", True, True, False)
        def measure(cache):
            crop_rng = random.Random(19)
            started = time.perf_counter()
            for _ in range(iterations):
                last = native_pair(sample, 1024, crop_rng, "depth", True, True, False, cache)
            return time.perf_counter() - started, last
        baseline, expected = measure(None)
        cache = DecodedImageCache(256 * 1024 * 1024)
        cached, actual = measure(cache)
        torch.testing.assert_close(actual[0], expected[0], atol=0, rtol=0)
        torch.testing.assert_close(actual[1], expected[1], atol=0, rtol=0)
        if actual[2] != expected[2]:
            raise ValueError("The optimization changed the random crop sequence.")
        return {"source_size": 2048, "crop_size": 1024, "iterations": iterations,
                "uncached_seconds": baseline, "cached_seconds": cached,
                "pixel_preparation_speedup": baseline / cached,
                "cache": cache.statistics(), "exact_final_pixels_and_crop_match": True,
                "scope": "CPU pixel preparation only, not complete training throughput"}


if __name__ == "__main__":
    print(json.dumps(benchmark(), indent=2))
