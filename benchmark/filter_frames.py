"""Automatic PSNR filtering (SVHighlights, Section 3.3.3).

Highlight videos occasionally contain frames that do not appear in the
corresponding full video (sponsor logos, intro/outro sequences, graphical
overlays). Such mismatched frames receive a low PSNR score during alignment.

A PSNR score below 20 is commonly regarded as indicating significant
perceptual dissimilarity, so this step removes every aligned frame whose PSNR
falls below the threshold (replacing its index with ``null``).

The paper additionally applies a lightweight manual filtering pass on top of
this automatic step; see the repository README for details.

Input : the directory of per-video alignment JSON files produced by align.py.
Output: a single JSON mapping each video name to a list of frame indices,
        with ``null`` for filtered-out highlight clips.
"""
import argparse
import json
import os

from natsort import natsorted

SPORTS = ['american_football', 'baseball', 'basketball', 'ice_hockey',
          'race', 'rugby', 'soccer', 'volleyball']


def filter_matching_result(file_path, psnr_threshold):
    """Keep frame indices with PSNR >= threshold, replace the rest with None."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    frames = []
    none_cnt = 0
    for item in data:
        if item['psnr'] >= psnr_threshold:
            frames.append(item["frame_idx"])
        else:
            frames.append(None)
            none_cnt += 1
    return frames, none_cnt, len(data)


def main():
    parser = argparse.ArgumentParser(
        description="Automatic PSNR filtering of aligned frames (Section 3.3.3).")
    parser.add_argument("--alignment_dir", type=str, required=True,
                        help="Directory of per-video alignment JSON files (align.py output).")
    parser.add_argument("--output", type=str, required=True,
                        help="Path of the aggregated filtered-frame-index JSON file.")
    parser.add_argument("--psnr_threshold", type=float, default=20.0,
                        help="Frames with PSNR below this value are filtered out (default: 20).")
    parser.add_argument("--sports", nargs="+", default=SPORTS, choices=SPORTS,
                        help="Sports to process (default: all).")
    args = parser.parse_args()

    all_jsons = natsorted(f for f in os.listdir(args.alignment_dir) if f.endswith(".json"))
    results = {}

    for sport in args.sports:
        json_files = [j for j in all_jsons if j.startswith(sport)]
        if not json_files:
            continue

        total_frames = 0
        remaining_frames = 0
        for file_name in json_files:
            file_path = os.path.join(args.alignment_dir, file_name)
            frames, none_cnt, total_frame_num = filter_matching_result(
                file_path, args.psnr_threshold)
            results[os.path.splitext(file_name)[0]] = frames
            total_frames += total_frame_num
            remaining_frames += (total_frame_num - none_cnt)

        if total_frames:
            print(f"{sport} remaining frame rate: {remaining_frames / total_frames:.4f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"Saved filtered frame indices to {args.output}")


if __name__ == "__main__":
    main()
