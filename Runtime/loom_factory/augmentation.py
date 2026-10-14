"""Independent, reproducible crop and transform timing for each source pair."""

import hashlib
import random


class AugmentationPlan:
    def __init__(self, seed, crop_refresh, augmentation_refresh):
        self.seed = seed
        self.crop_refresh = crop_refresh
        self.augmentation_refresh = augmentation_refresh
        self.crop_visits = 0
        self.augmentation_visits = 0

    def _rng(self, stream, refresh, sample_id, epoch, visit):
        period = 0 if refresh == "per_run" else epoch
        occurrence = visit if refresh == "per_sample" else 0
        key = f"{self.seed}|{stream}|{sample_id}|{period}|{occurrence}".encode()
        return random.Random(int.from_bytes(hashlib.sha256(key).digest(), "big"))

    def generators(self, sample_id, epoch):
        crop = self._rng("crop", self.crop_refresh, sample_id, epoch, self.crop_visits)
        transform = self._rng("transform", self.augmentation_refresh, sample_id, epoch, self.augmentation_visits)
        self.crop_visits += 1
        self.augmentation_visits += 1
        return crop, transform
