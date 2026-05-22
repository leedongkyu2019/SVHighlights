"""Automatic speech recognition (SVHighlights, Section 4.1, Stage 1).

TF-SELECTOR uses transcript information to merge visually-distinct shots that
share the same spoken sentence into a single context-aware segment. We obtain
word-level transcripts with WhisperX.

For each input video this script writes two JSON files:

  * ``--whisper_dir/<video>.json``      -- flat list of words, each with
        {"idx", "start", "end", "word"} (word-level timestamps, used by
        segment.py).
  * ``--whisper_all_dir/<video>.json``  -- the raw WhisperX segment objects.
"""
import argparse
import json
import os
import warnings

from natsort import natsorted

import whisperx

warnings.filterwarnings("ignore")

SPORTS = ['american_football', 'baseball', 'basketball', 'ice_hockey',
          'race', 'rugby', 'soccer', 'volleyball']


def transcribe(video_path_list, whisper_path_list, whisper_all_path_list, args):
    model = whisperx.load_model(args.model, args.device,
                                compute_type=args.compute_type, language=args.language)

    for idx, video_path in enumerate(video_path_list):
        audio = whisperx.load_audio(video_path)
        result = model.transcribe(audio, batch_size=args.batch_size)
        model_a, metadata = whisperx.load_align_model(
            language_code=result["language"], device=args.device)
        result = whisperx.align(result["segments"], model_a, metadata, audio,
                                args.device, return_char_alignments=False)

        segments = result['segments']
        words = []
        for s in segments:
            for w in s['words']:
                words.append({
                    "idx": len(words),
                    "start": float(w['start']),
                    "end": float(w['end']),
                    "word": w['word'],
                })

        with open(whisper_path_list[idx], 'w') as f:
            json.dump(words, f, indent=4)
        with open(whisper_all_path_list[idx], 'w') as f:
            json.dump(segments, f, indent=4)
        print(f"Progress: {idx + 1} / {len(video_path_list)}")


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe videos with WhisperX (Section 4.1).")
    parser.add_argument("--video_dir", type=str, required=True,
                        help="Directory containing the input videos.")
    parser.add_argument("--whisper_dir", type=str, required=True,
                        help="Output directory for word-level transcripts.")
    parser.add_argument("--whisper_all_dir", type=str, required=True,
                        help="Output directory for raw WhisperX segments.")
    parser.add_argument("--sports", nargs="+", default=SPORTS, choices=SPORTS,
                        help="Sports to process (default: all).")
    parser.add_argument("--model", type=str, default="large-v2",
                        help="WhisperX model name.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Reduce if low on GPU memory.")
    parser.add_argument("--compute_type", type=str, default="float16",
                        help="Use 'int8' if low on GPU memory (may reduce accuracy).")
    parser.add_argument("--language", type=str, default="en")
    args = parser.parse_args()

    os.makedirs(args.whisper_dir, exist_ok=True)
    os.makedirs(args.whisper_all_dir, exist_ok=True)

    video_path_list, whisper_path_list, whisper_all_path_list = [], [], []
    for f in natsorted(os.listdir(args.video_dir)):
        if not (f.endswith(".mp4") and any(f.startswith(s) for s in args.sports)):
            continue
        whisper_path = os.path.join(args.whisper_dir, f.replace(".mp4", ".json"))
        if os.path.isfile(whisper_path):
            print(f"{whisper_path} already exists, skipping.")
            continue
        video_path_list.append(os.path.join(args.video_dir, f))
        whisper_path_list.append(whisper_path)
        whisper_all_path_list.append(
            os.path.join(args.whisper_all_dir, f.replace(".mp4", ".json")))

    if video_path_list:
        transcribe(video_path_list, whisper_path_list, whisper_all_path_list, args)
    print(f"Done. Transcribed {len(video_path_list)} video(s).")


if __name__ == "__main__":
    main()
