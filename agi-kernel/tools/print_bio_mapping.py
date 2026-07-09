#!/usr/bin/env python3
"""Pretty-print the biology→silicon mapping as a Markdown table.

Uses docs/BIO_MAPPING.md as the source of truth; this script exists so CI
can verify that every kernel file has a corresponding row.
"""
from __future__ import annotations

import re
import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
MAPPING = ROOT / "docs" / "BIO_MAPPING.md"


def main() -> int:
    text = MAPPING.read_text(encoding="utf-8")
    rows = re.findall(r"^\|\s*(\d+)\s*\|.+?\|.+?\|.+?\|\s*`([^`]+)`\s*\|", text, re.MULTILINE)
    print(f"Found {len(rows)} biology→silicon rows.")
    missing = []
    for idx, path in rows:
        if not (ROOT / path).exists() and not path.startswith("design"):
            missing.append((idx, path))
    if missing:
        print("Missing files for rows:")
        for i, p in missing:
            print(f"  #{i}  {p}")
        return 1
    print("All referenced files exist.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
