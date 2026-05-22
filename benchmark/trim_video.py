"""Video trimming (SVHighlights, Section 3.2).

Full-length broadcasts often contain segments unrelated to the target game
(intros, footage from previous matches, post-game interviews). This script
trims each full video down to the actual game footage using the per-video
start/end times listed in a metadata CSV.

The boundary times are obtained from a single boundary judgment per video,
following the per-sport game start/end cues described in the paper
(Appendix Table 11).

Metadata CSV format -- the dataset's ``video_list.csv`` (one row per video):
    vid,full_link,full_length,full_start,full_end,hl_link,hl_length
This script uses ``vid`` and the trim boundaries ``full_start`` / ``full_end``
(in seconds); the remaining columns are ignored.
"""
import argparse
import os
import re
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

SPORTS = ['american_football', 'baseball', 'basketball', 'ice_hockey',
          'race', 'rugby', 'soccer', 'volleyball']


def cut(vid_path, output_path, start, end):
    """Trim ``vid_path`` to the [start, end] interval (seconds) with ffmpeg."""
    try:
        command = [
            'ffmpeg',
            '-ss', str(start),
            '-to', str(end),
            '-i', vid_path,
            '-c:v', 'libx264',
            '-c:a', 'aac',
            '-preset', 'fast',
            '-threads', '2',
            '-y',
            output_path,
        ]
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"vid_path: {vid_path}\nerr: {e}")


def extract_number(file_name):
    """Extract the numeric id from a file name (e.g. 'soccer_5' -> '5')."""
    match = re.search(r'\d+', str(file_name))
    return match.group() if match else None


def main():
    parser = argparse.ArgumentParser(
        description="Trim full-length videos to actual game footage (Section 3.2).")
    parser.add_argument("--video_list", type=str, required=True,
                        help="Dataset video_list.csv (uses columns: vid, full_start, full_end).")
    parser.add_argument("--src_dir", type=str, required=True,
                        help="Directory containing the raw (untrimmed) full videos.")
    parser.add_argument("--dst_dir", type=str, required=True,
                        help="Directory to write the trimmed full videos.")
    parser.add_argument("--sports", nargs="+", default=SPORTS, choices=SPORTS,
                        help="Sports to process (default: all).")
    parser.add_argument("--idx", type=str, default=None,
                        help="Optional single video id to process (e.g. '5').")
    parser.add_argument("--max_workers", type=int, default=4,
                        help="Number of parallel ffmpeg workers.")
    args = parser.parse_args()

    os.makedirs(args.dst_dir, exist_ok=True)
    video_list = pd.read_csv(args.video_list)

    preprocess_list = []
    for _, row in video_list.iterrows():
        vid = row["vid"]
        if not any(vid.startswith(sport) for sport in args.sports):
            continue
        if args.idx is not None and extract_number(vid) != args.idx:
            continue

        vid_path = os.path.join(args.src_dir, vid + ".mp4")
        output_path = os.path.join(args.dst_dir, vid + ".mp4")
        if not os.path.exists(vid_path):
            print(f"[skip] source not found: {vid_path}")
            continue
        if os.path.exists(output_path):
            print(f"[skip] already trimmed: {output_path}")
            continue
        preprocess_list.append((vid_path, output_path, row["full_start"], row["full_end"]))

    print(f"Trimming {len(preprocess_list)} video(s)...")
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [
            executor.submit(cut, vid_path, output_path, start, end)
            for vid_path, output_path, start, end in preprocess_list
        ]
        for future in as_completed(futures):
            future.result()


if __name__ == "__main__":
    main()
