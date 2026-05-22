"""Context-aware video segmentation (SVHighlights, Section 4.1, Stage 1).

Individual shots are often too short to be meaningful semantic units. This
script merges adjacent shots that share the same spoken sentence into
context-aware segments, which serve as the basic unit of TF-SELECTOR.

  * Consecutive transcribed words less than one second apart are grouped into
    a sentence.
  * When a sentence spans two adjacent shots, those shots are merged.
  * To keep segments from becoming too long, a maximum segment-length
    constraint is imposed (the paper sets it to 2 minutes); shots that would
    exceed this limit are split instead of merged.

Inputs : word-level transcripts (transcribe.py), shot boundaries
         (shot_boundary.py), and the videos themselves (for fps).
Output : one JSON file per video under ``--output_dir``, a list of segments
         each with {"idx", "start", "end", "text", "segment_interval",
         "duration"}.
"""
import argparse
import json
import os

from natsort import natsorted

SPORTS = ['american_football', 'baseball', 'basketball', 'ice_hockey',
          'race', 'rugby', 'soccer', 'volleyball']


def seconds_to_minutes_seconds(seconds):
    """Format a number of seconds as a 'minutes:seconds' string."""
    minutes = seconds // 60
    remaining_seconds = seconds % 60
    return f"{int(minutes)}:{int(remaining_seconds):02d}"


def load_json(json_file_path):
    with open(json_file_path, 'r') as f:
        return json.load(f)


def get_video_fps(video_path):
    """Return the fps of the first video stream using ffprobe."""
    import subprocess
    command = [
        "ffprobe", "-v", "0", "-of", "json",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        video_path,
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, text=True)
    ffprobe_data = json.loads(result.stdout)
    num, den = ffprobe_data["streams"][0]["r_frame_rate"].split("/")
    return float(num) / float(den)


def read_shot_boundaries(file_path, fps):
    """Read a shot-boundary file (start/end frame indices) into time intervals."""
    shots = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                frame_start, frame_end = int(parts[0]), int(parts[1])
                shots.append({"start": frame_start / fps, "end": frame_end / fps})
    return shots


def split_shots_equally(shots, max_duration):
    """Split shots longer than max_duration into equal-length sub-shots."""
    split_shots = []
    for shot in shots:
        start_time = shot["start"]
        end_time = shot["end"]
        shot_duration = end_time - start_time

        if shot_duration <= max_duration:
            split_shots.append({"start": start_time, "end": end_time})
        else:
            num_splits = int(shot_duration // max_duration) + 1
            split_length = shot_duration // num_splits
            remainder = shot_duration % num_splits
            for i in range(num_splits):
                new_start = start_time + (i * split_length)
                new_end = new_start + split_length
                if i == num_splits - 1:
                    new_end += remainder
                split_shots.append({"start": new_start, "end": new_end})
    return split_shots


def merge_shots_with_whisper(shots, whispers, output_path, max_duration):
    """Align Whisper sentences with shots and merge them into segments."""
    shots = split_shots_equally(shots, max_duration)
    merged_shots = [{"idx": 0, "start": shots[0]['start'], "end": shots[0]['end'], "text": ""}]
    next_shot_idx = 1

    for w in whispers:
        try:
            # Start a new segment whenever the sentence begins after the
            # current one ends.
            while w['start'] >= merged_shots[-1]['end']:
                merged_shots.append({
                    "idx": len(merged_shots),
                    "start": shots[next_shot_idx]['start'],
                    "end": shots[next_shot_idx]['end'],
                    "text": "",
                })
                next_shot_idx += 1

            # Extend the current segment, respecting the max-duration limit.
            while w['end'] > merged_shots[-1]['end']:
                current_duration = float(merged_shots[-1]['end']) - float(merged_shots[-1]['start'])
                next_shot_len = float(shots[next_shot_idx]['end']) - float(shots[next_shot_idx]['start'])

                # If merging the next shot would exceed max_duration, flush the
                # words that already fit and open a new segment.
                if current_duration + next_shot_len > max_duration:
                    new_words = []
                    for word in w['text']:
                        if word['end'] <= merged_shots[-1]['end']:
                            merged_shots[-1]['text'] = (
                                word['word'] if merged_shots[-1]['text'] == ""
                                else merged_shots[-1]['text'] + " " + word['word'])
                        else:
                            new_words.append(word)
                    w['text'] = new_words

                    merged_shots.append({
                        "idx": len(merged_shots),
                        "start": shots[next_shot_idx]['start'],
                        "end": shots[next_shot_idx]['end'],
                        "text": "",
                    })

                merged_shots[-1]['end'] = shots[next_shot_idx]['end']
                next_shot_idx += 1
        except IndexError:
            # No shots left; remaining words fall into the last segment.
            pass

        for word in w['text']:
            merged_shots[-1]['text'] = (
                word['word'] if merged_shots[-1]['text'] == ""
                else merged_shots[-1]['text'] + " " + word['word'])

    # Append any trailing shots that received no transcript.
    while next_shot_idx < len(shots):
        merged_shots.append({
            "idx": len(merged_shots),
            "start": shots[next_shot_idx]['start'],
            "end": shots[next_shot_idx]['end'],
            "text": "",
        })
        next_shot_idx += 1

    for s in merged_shots:
        s['segment_interval'] = (f"{seconds_to_minutes_seconds(s['start'])}"
                                 f"~{seconds_to_minutes_seconds(s['end'])}")
        s['duration'] = seconds_to_minutes_seconds(float(s['end']) - float(s['start']))
        s['start'] = f"{s['start']:.3f}"
        s['end'] = f"{s['end']:.3f}"

    with open(output_path, 'w') as f:
        json.dump(merged_shots, f, indent=4)
    print(f"Segment results saved to: {output_path}")


def build_sentences(whisper_data):
    """Group word-level transcripts into sentences (gap < 1 second)."""
    merged_whisper = []
    for word in whisper_data:
        current_start = float(word['start'])
        current_end = float(word['end'])
        if not merged_whisper:
            merged_whisper.append({"idx": 0, "start": current_start,
                                   "end": current_end, "text": [word]})
        else:
            last_end = float(merged_whisper[-1]['end'])
            if abs(current_start - last_end) <= 1:
                merged_whisper[-1]['end'] = current_end
                merged_whisper[-1]['text'].append(word)
            else:
                merged_whisper.append({"idx": len(merged_whisper), "start": current_start,
                                       "end": current_end, "text": [word]})

    for w in merged_whisper:
        w['segment_interval'] = (f"{seconds_to_minutes_seconds(w['start'])}"
                                 f"~{seconds_to_minutes_seconds(w['end'])}")
        w['duration'] = seconds_to_minutes_seconds(float(w['end']) - float(w['start']))
    return merged_whisper


def main():
    parser = argparse.ArgumentParser(
        description="Merge shots and transcripts into context-aware segments (Section 4.1).")
    parser.add_argument("--whisper_dir", type=str, required=True,
                        help="Directory of word-level transcripts (transcribe.py output).")
    parser.add_argument("--video_dir", type=str, required=True,
                        help="Directory of the videos (used to read fps).")
    parser.add_argument("--shot_dir", type=str, required=True,
                        help="Directory of shot-boundary files (shot_boundary.py output).")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to write per-video segment JSON files.")
    parser.add_argument("--merged_whisper_dir", type=str, default=None,
                        help="Optional directory to also save grouped sentence transcripts.")
    parser.add_argument("--max_duration", type=float, default=120.0,
                        help="Maximum segment length in seconds (paper uses 120).")
    parser.add_argument("--sports", nargs="+", default=SPORTS, choices=SPORTS,
                        help="Sports to process (default: all).")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if args.merged_whisper_dir:
        os.makedirs(args.merged_whisper_dir, exist_ok=True)

    for sport in args.sports:
        whisper_files = natsorted(
            f for f in os.listdir(args.whisper_dir)
            if f.startswith(sport) and f.endswith(".json"))

        for whisper_name in whisper_files:
            stem = os.path.splitext(whisper_name)[0]
            video_path = os.path.join(args.video_dir, stem + ".mp4")
            shot_path = os.path.join(args.shot_dir, stem + ".mp4.scenes.txt")

            if not os.path.exists(video_path):
                print(f"[skip] video not found: {video_path}")
                continue
            if not os.path.exists(shot_path):
                print(f"[skip] shot file not found: {shot_path}")
                continue

            whisper_data = load_json(os.path.join(args.whisper_dir, whisper_name))
            merged_whisper = build_sentences(whisper_data)

            if args.merged_whisper_dir:
                with open(os.path.join(args.merged_whisper_dir, whisper_name), 'w') as f:
                    json.dump(merged_whisper, f, indent=4)

            fps = get_video_fps(video_path)
            shots = read_shot_boundaries(shot_path, fps)
            output_path = os.path.join(args.output_dir, whisper_name)
            merge_shots_with_whisper(shots, merged_whisper, output_path, args.max_duration)


if __name__ == "__main__":
    main()
