"""Shot boundary detection (SVHighlights, Section 4.1, Stage 1).

TF-SELECTOR's context-aware segmentation first splits each video into
individual shots. We use TransNet V2 [Soucek and Lokoc, 2024] as the
shot-boundary detector.

TransNet V2 is a third-party tool and is NOT bundled with this repository.
Clone it first:

    git clone https://github.com/soCzech/TransNetV2

then point ``--transnetv2_script`` at its inference entrypoint, e.g.

    TransNetV2/inference/transnetv2.py            (TensorFlow)
    TransNetV2/inference-pytorch/transnetv2.py    (PyTorch)

For each input video the detector writes ``<video>.scenes.txt`` (shot
boundaries as start/end frame-index pairs) and ``<video>.predictions.txt``
next to the video. This script then moves those files into ``--output_dir``.
"""
import argparse
import os
import shutil
import subprocess
import sys

from natsort import natsorted

SPORTS = ['american_football', 'baseball', 'basketball', 'ice_hockey',
          'race', 'rugby', 'soccer', 'volleyball']


def main():
    parser = argparse.ArgumentParser(
        description="Detect shot boundaries with TransNet V2 (Section 4.1).")
    parser.add_argument("--video_dir", type=str, required=True,
                        help="Directory containing the input videos.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to move the generated shot-boundary files into.")
    parser.add_argument("--transnetv2_script", type=str, required=True,
                        help="Path to the TransNet V2 inference script "
                             "(see module docstring for setup instructions).")
    parser.add_argument("--sports", nargs="+", default=SPORTS, choices=SPORTS,
                        help="Sports to process (default: all).")
    args = parser.parse_args()

    if not os.path.isfile(args.transnetv2_script):
        sys.exit(f"TransNet V2 script not found: {args.transnetv2_script}")

    video_paths = natsorted(
        os.path.join(args.video_dir, f)
        for f in os.listdir(args.video_dir)
        if f.endswith(".mp4") and any(f.startswith(s) for s in args.sports))

    if not video_paths:
        sys.exit(f"No matching videos found in {args.video_dir}")

    print(f"Running TransNet V2 on {len(video_paths)} video(s)...")
    subprocess.run([sys.executable, args.transnetv2_script] + video_paths, check=True)

    os.makedirs(args.output_dir, exist_ok=True)
    for filename in os.listdir(args.video_dir):
        if filename.endswith(".txt"):
            src_path = os.path.join(args.video_dir, filename)
            dst_path = os.path.join(args.output_dir, filename)
            shutil.move(src_path, dst_path)
            print(f"Moved {src_path} -> {dst_path}")


if __name__ == "__main__":
    main()
