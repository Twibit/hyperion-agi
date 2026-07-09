"""
Biological Data Pipeline — V14 Aurora-Foundation.
===================================================
A real biological brain does not train on "epochs" of static datasets.
It experiences a continuous, infinite stream of sensory data.

This module provides an infinite generator that can read massive JSONL
files (like Wikipedia dumps or code corpora) lazily, feeding the organism
one sentence or image at a time without crashing memory.
"""
import json
import gzip
import os
from typing import Generator, Dict, Any

class InfiniteSensoryStream:
    def __init__(self, filepath: str = None):
        self.filepath = filepath
        
    def _read_file_lazily(self) -> Generator[str, None, None]:
        """Yields raw lines from a compressed or plain text file."""
        if self.filepath.endswith('.gz'):
            with gzip.open(self.filepath, 'rt', encoding='utf-8') as f:
                for line in f:
                    yield line
        else:
            with open(self.filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    yield line

    def stream_text(self) -> Generator[Dict[str, Any], None, None]:
        """
        Infinite stream of text observations.
        If the file ends, it loops back to the beginning to simulate
        continuous environmental exposure.
        """
        if self.filepath and os.path.exists(self.filepath):
            while True:
                for line in self._read_file_lazily():
                    line = line.strip()
                    if not line:
                        continue
                        
                    # Handle JSONL format (e.g. {"text": "The cat sat..."})
                    if line.startswith("{"):
                        try:
                            data = json.loads(line)
                            if "text" in data:
                                yield {"text": data["text"]}
                        except json.JSONDecodeError:
                            yield {"text": line}
                    else:
                        yield {"text": line}
        else:
            # Fallback mock infinite environment if no massive dataset is provided
            environments = [
                "The cat sat on the mat.",
                "A dog barked loudly outside.",
                "Rain falls from the gray clouds.",
                "The server processed 1000 requests per second.",
                "def calculate_atp(fatigue): return max(0, 100 - fatigue)"
            ]
            idx = 0
            while True:
                yield {"text": environments[idx]}
                idx = (idx + 1) % len(environments)

    def stream_vision(self) -> Generator[Dict[str, Any], None, None]:
        """
        Mock infinite stream of visual embeddings.
        In a production environment, this would yield frames from a webcam
        or batches from a video dataset, already embedded by a visual cortex (e.g., CLIP).
        """
        import numpy as np
        rng = np.random.default_rng(42)
        while True:
            # Simulated 512-dim visual embedding from an optic nerve
            simulated_optic_nerve = rng.standard_normal(512, dtype=np.float32)
            yield {"vision": simulated_optic_nerve}
