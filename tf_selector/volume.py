"""Audio volume extraction (SVHighlights, Section 4.3, Stage 3).

TF-SELECTOR's segment-level scoring uses audio volume as one of its three
input modalities -- loud moments (crowd reactions, commentary emphasis) are a
useful cue for highlights.

This script computes the loudness (dBFS) of every non-overlapping 2-second
clip of each video and stores it as a per-sport JSON file mapping each video
name to a list of per-clip volume values.
"""
import argparse
import json
import os

from natsort import natsorted
from pydub import AudioSegment
from pydub.utils import make_chunks

SPORTS = ['american_football', 'baseball', 'basketball', 'ice_hockey',
          'race', 'rugby', 'soccer', 'volleyball']


def clip_volumes(video_path, chunk_ms):
    """Return the dBFS loudness of every ``chunk_ms`` clip of a video."""
    audio = AudioSegment.from_file(video_path, format="mp4")
    chunks = make_chunks(audio, chunk_ms)
    return [round(float(chunk.dBFS), 2) for chunk in chunks]


def main():
    parser = argparse.ArgumentParser(
        description="Extract per-clip audio volume (Section 4.3).")
    parser.add_argument("--video_dir", type=str, required=True,
                        help="Directory containing the input videos.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to write per-sport volume JSON files.")
    parser.add_argument("--sports", nargs="+", default=SPORTS, choices=SPORTS,
                        help="Sports to process (default: all).")
    parser.add_argument("--chunk_ms", type=int, default=2000,
                        help="Clip length in milliseconds (paper uses 2000 = 2s).")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for sport in args.sports:
        videos = natsorted(
            f for f in os.listdir(args.video_dir)
            if f.startswith(sport) and f.endswith(".mp4"))
        if not videos:
            continue

        volumes = {}
        output_path = os.path.join(args.output_dir, f"{sport}.json")
        for i, video_name in enumerate(videos, start=1):
            vid = os.path.splitext(video_name)[0]
            volumes[vid] = clip_volumes(os.path.join(args.video_dir, video_name),
                                        args.chunk_ms)
            print(f"[{sport}] {i}/{len(videos)} {vid}")
            if i % 10 == 0:  # periodic checkpoint
                with open(output_path, "w") as f:
                    json.dump(volumes, f)

        with open(output_path, "w") as f:
            json.dump(volumes, f)
        print(f"Saved volumes to {output_path}")


if __name__ == "__main__":
    main()
