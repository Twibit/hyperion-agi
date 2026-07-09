"""ZPD self-paced sampler — identical to v2."""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List, Sequence


@dataclass
class CurriculumStats:
    accepted: int = 0
    rejected_too_easy: int = 0
    rejected_too_hard: int = 0


class ZPDCurriculum:
    def __init__(self, lower, upper):
        self.lower, self.upper = lower, upper
        self.stats = CurriculumStats()

    def filter(self, items, error_fn):
        keep = []
        for it in items:
            e = float(error_fn(it))
            if e < self.lower: self.stats.rejected_too_easy += 1
            elif e > self.upper: self.stats.rejected_too_hard += 1
            else: self.stats.accepted += 1; keep.append(it)
        return keep

    def adapt(self, mean_error):
        target = max(0.05, mean_error * 0.6)
        self.lower = 0.9 * self.lower + 0.1 * target
        self.upper = 0.9 * self.upper + 0.1 * (target * 3.0)
