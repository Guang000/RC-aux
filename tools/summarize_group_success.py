#!/usr/bin/env python3
import argparse
import glob
import re
from pathlib import Path

import numpy as np


def extract_success_rate(path: Path) -> float:
    text = path.read_text()
    match = re.search(r"success_rate':\s*([0-9.]+)", text)
    if not match:
        raise ValueError(f"Could not parse success_rate from {path}")
    return float(match.group(1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args()

    rows = []
    for raw in args.paths:
        for path_str in sorted(glob.glob(raw)):
            path = Path(path_str)
            rows.append((str(path), extract_success_rate(path)))

    if not rows:
        raise SystemExit("No files matched.")

    values = np.array([v for _, v in rows], dtype=float)
    for path, value in rows:
        print(f"{path}\t{value:.1f}")
    print(f"mean\t{values.mean():.2f}")
    print(f"std\t{values.std(ddof=0):.2f}")
    print(f"min\t{values.min():.1f}")
    print(f"max\t{values.max():.1f}")


if __name__ == "__main__":
    main()
