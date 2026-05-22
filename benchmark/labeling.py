"""Highlight label generation (SVHighlights, Section 3.4).

Converts the aligned highlight frames into the final per-clip highlight labels
that constitute the SVHighlights benchmark ground truth.

The full video is divided into non-overlapping 2-second clips. Each aligned
highlight frame is turned into a 1-second highlighted segment (0.5s before and
after the frame). A 2-second clip is labeled 1 (highlight) if at least 50% of
its duration -- i.e. >= 1.0s -- overlaps the highlighted segments, and 0
otherwise.

Inputs:
  * the filtered-frame-index JSON produced by filter_frames.py
  * the trimmed 144p full videos (to map frame indices to real timestamps)
Output: a JSON list of {"vid", "saliency_scores": [0/1, ...]}.

Based on the previous-submission script ``code/benchmark/labeling.py``;
cleaned up (CLI arguments, no hard-coded paths, the ffprobe frame-timestamp
field replaced with the version-robust ``best_effort_timestamp_time`` -- the
old ``pkt_pts_time`` is removed in recent ffmpeg, while ``pts_time`` is absent
in older ffmpeg) and the dependency on a separately-sampled frame folder
removed -- the number of 2-second clips is now derived directly from the
video duration.
"""
import argparse
import json
import os
import subprocess

from natsort import natsorted

SPORTS = ['american_football', 'baseball', 'basketball', 'ice_hockey',
          'race', 'rugby', 'soccer', 'volleyball']


def get_frame_timestamps(video_path):
    """Return the presentation timestamp (seconds) of every frame."""
    cmd = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'frame=best_effort_timestamp_time',
        '-of', 'csv=p=0', video_path,
    ]
    try:
        output = subprocess.check_output(cmd).decode().strip().split('\n')
        return [float(ts) for ts in output if ts]
    except subprocess.CalledProcessError as e:
        print(f"ffprobe error on {video_path}: {e}")
        return []


def get_video_duration(video_path):
    """Return the duration of a video in seconds."""
    cmd = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=duration',
        '-of', 'csv=p=0', video_path,
    ]
    try:
        return float(subprocess.check_output(cmd).decode().strip())
    except (subprocess.CalledProcessError, ValueError) as e:
        print(f"ffprobe error on {video_path}: {e}")
        return 0.0


def label_clips(clip_starts, gt_segments, clip_len=2.0, overlap_ratio=0.5):
    """Assign a 0/1 label to each clip.

    A clip is labeled 1 when its total overlap with the highlighted segments
    reaches ``clip_len * overlap_ratio`` seconds (1.0s for a 2-second clip at
    a 50% ratio).
    """
    threshold = clip_len * overlap_ratio
    labels = []
    for start in clip_starts:
        end = start + clip_len
        overlap = 0.0
        for gt_start, gt_end in gt_segments:
            o = min(end, gt_end) - max(start, gt_start)
            if o > 0:
                overlap += o
        labels.append(1 if overlap >= threshold else 0)
    return labels


def main():
    parser = argparse.ArgumentParser(
        description="Generate per-clip highlight labels (Section 3.4).")
    parser.add_argument("--filtered_json", type=str, required=True,
                        help="Filtered-frame-index JSON (filter_frames.py output).")
    parser.add_argument("--video_dir", type=str, required=True,
                        help="Directory of trimmed full-length videos (144p).")
    parser.add_argument("--output", type=str, required=True,
                        help="Path of the output label JSON file.")
    parser.add_argument("--sports", nargs="+", default=SPORTS, choices=SPORTS,
                        help="Sports to process (default: all).")
    parser.add_argument("--clip_len", type=float, default=2.0,
                        help="Clip length in seconds (paper uses 2).")
    parser.add_argument("--overlap_ratio", type=float, default=0.5,
                        help="Minimum overlap ratio for a positive label (paper uses 0.5).")
    args = parser.parse_args()

    with open(args.filtered_json, "r", encoding="utf-8") as f:
        matching_result = json.load(f)

    vids = natsorted(k for k in matching_result
                     if any(k.startswith(s) for s in args.sports))

    full_labels = []
    for vid in vids:
        video_path = os.path.join(args.video_dir, f"{vid}.mp4")
        if not os.path.exists(video_path):
            print(f"[skip] video not found: {video_path}")
            continue

        frame_timestamps = get_frame_timestamps(video_path)
        if not frame_timestamps:
            print(f"[skip] could not read frame timestamps: {vid}")
            continue
        duration = get_video_duration(video_path)

        # Non-overlapping 2-second clips spanning the full video.
        clip_starts = list(range(0, int(duration), int(args.clip_len)))

        # Each aligned (non-filtered) highlight frame -> a 1-second segment.
        gt_segments = []
        for middle_frame in matching_result[vid]:
            if middle_frame is None:               # filtered out
                continue
            if middle_frame >= len(frame_timestamps):
                continue
            matched_time = frame_timestamps[middle_frame]
            gt_segments.append((matched_time - 0.5, matched_time + 0.5))

        labels = label_clips(clip_starts, gt_segments,
                             clip_len=args.clip_len, overlap_ratio=args.overlap_ratio)
        full_labels.append({"vid": vid, "saliency_scores": labels})
        print(f"{vid}: {len(labels)} clips, {sum(labels)} positive")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(full_labels, f)
    print(f"Saved {len(full_labels)} videos' labels to {args.output}")


if __name__ == "__main__":
    main()
