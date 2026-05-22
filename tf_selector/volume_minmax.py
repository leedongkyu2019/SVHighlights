"""Volume normalization (TF-SELECTOR, Section 4.3, Stage 3).

``volume.py`` outputs raw loudness in dBFS. This script normalizes those
values to the [0, 1] range used by the segment-level scoring LLM: dBFS is
treated as spanning [-100, 0] dB, silent clips (-inf dBFS) are clamped to
-100, and each value v is mapped to (v + 100) / 100.

Input : the directory of per-sport volume JSON files produced by volume.py.
Output: a single combined JSON mapping every video name to its normalized
        per-clip volume list.
"""
import argparse
import json
import math
import os

from natsort import natsorted


def normalize(value):
    """Map a dBFS value into [0, 1]; silent clips (-inf) clamp to 0."""
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        value = -100.0  # -inf / "-Inf" -> silence
    value = min(max(value, -100.0), 0.0)
    return round((value + 100.0) / 100.0, 2)


def main():
    parser = argparse.ArgumentParser(
        description="Normalize per-clip audio volume to [0, 1] (Section 4.3).")
    parser.add_argument("--volume_dir", type=str, required=True,
                        help="Directory of per-sport volume JSON files (volume.py output).")
    parser.add_argument("--output", type=str, required=True,
                        help="Path of the combined normalized volume JSON file.")
    args = parser.parse_args()

    combined = {}
    for fname in natsorted(os.listdir(args.volume_dir)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(args.volume_dir, fname), "r") as f:
            volume = json.load(f)
        for vid, values in volume.items():
            combined[vid] = [normalize(v) for v in values]

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(combined, f)
    print(f"Saved normalized volumes for {len(combined)} videos to {args.output}")


if __name__ == "__main__":
    main()
