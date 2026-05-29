"""CLIP-similarity evaluation (SVHighlights, Section 3.3, Table 3).

Reports the average CLIP (ViT-B/32) cosine similarity between every aligned
highlight frame and its matched full-video frame -- a feature-level
complement to the pixel-level PSNR/SSIM metrics in eval_matching_quality.py.

Input : the filtered-frame-index JSON produced by filter_frames.py.
Output: one JSON per sport under ``--output_dir`` mapping each video name to
        its mean CLIP similarity.

Requires the OpenAI CLIP package:
    pip install git+https://github.com/openai/CLIP.git
"""
import argparse
import json
import os
import subprocess

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

import clip

SPORTS = ['american_football', 'baseball', 'basketball', 'ice_hockey',
          'race', 'rugby', 'soccer', 'volleyball']

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_model = None
_preprocess = None


def _load_clip():
    """Load the CLIP model on first use (cached), so importing this module
    and running ``--help`` do not touch the GPU."""
    global _model, _preprocess
    if _model is None:
        _model, _preprocess = clip.load("ViT-B/32", device=DEVICE)
    return _model, _preprocess


def clip_sim(full_video_vectors, highlight_vectors):
    """Mean cosine similarity of CLIP image features over aligned frame pairs."""
    model, preprocess = _load_clip()
    sum_sim, num = 0.0, 0
    for full_vector, highlight_vector in zip(full_video_vectors, highlight_vectors):
        full_image = preprocess(Image.fromarray(full_vector)).unsqueeze(0).to(DEVICE)
        highlight_image = preprocess(Image.fromarray(highlight_vector)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            full_feats = model.encode_image(full_image)
            highlight_feats = model.encode_image(highlight_image)
            full_feats /= full_feats.norm(dim=-1, keepdim=True)
            highlight_feats /= highlight_feats.norm(dim=-1, keepdim=True)
            sum_sim += (full_feats @ highlight_feats.t()).squeeze().item()
            num += 1
    return sum_sim / num if num else 0.0


def extract_all_frames(video_path):
    """Decode every frame of a video into a list of BGR numpy arrays."""
    probe_cmd = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,nb_frames,avg_frame_rate,duration',
        '-of', 'json', video_path,
    ]
    probe = subprocess.run(probe_cmd, stdout=subprocess.PIPE,
                           stderr=subprocess.DEVNULL, universal_newlines=True)
    stream = json.loads(probe.stdout)['streams'][0]
    width, height = int(stream['width']), int(stream['height'])
    nb_frames = int(stream.get('nb_frames', 0))

    command = ['ffmpeg', '-i', video_path, '-f', 'image2pipe',
               '-pix_fmt', 'bgr24', '-vcodec', 'rawvideo', '-']
    pipe = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    frames = []
    frame_size = width * height * 3
    with tqdm(total=nb_frames if nb_frames > 0 else None,
              desc=f"Extracting {os.path.basename(video_path)}") as pbar:
        while True:
            raw_frame = pipe.stdout.read(frame_size)
            if len(raw_frame) != frame_size:
                break
            frames.append(np.frombuffer(raw_frame, np.uint8).reshape((height, width, 3)))
            pbar.update(1)
    pipe.stdout.close()
    pipe.wait()
    return frames


def extract_middle_sampled_frames(video_path):
    """Decode the middle frame of every 1-second clip."""
    probe_cmd = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,duration,avg_frame_rate',
        '-of', 'json', video_path,
    ]
    probe = subprocess.run(probe_cmd, stdout=subprocess.PIPE,
                           stderr=subprocess.DEVNULL, universal_newlines=True)
    stream = json.loads(probe.stdout)['streams'][0]
    width, height = int(stream['width']), int(stream['height'])
    total_seconds = int(float(stream['duration']))

    command = ['ffmpeg', '-ss', '0.5', '-i', video_path, '-vf', 'fps=1',
               '-f', 'image2pipe', '-pix_fmt', 'bgr24', '-vcodec', 'rawvideo', '-']
    pipe = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    frames = []
    frame_size = width * height * 3
    with tqdm(total=total_seconds,
              desc=f"Extracting middle frames {os.path.basename(video_path)}") as pbar:
        while True:
            raw_frame = pipe.stdout.read(frame_size)
            if len(raw_frame) != frame_size:
                break
            frames.append(np.frombuffer(raw_frame, np.uint8).reshape((height, width, 3)))
            pbar.update(1)
    pipe.stdout.close()
    pipe.wait()
    return frames


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate alignment quality with CLIP similarity (Table 3).")
    parser.add_argument("--filtered_json", type=str, required=True,
                        help="Filtered-frame-index JSON (filter_frames.py output).")
    parser.add_argument("--full_dir", type=str, required=True,
                        help="Directory of full-length videos (144p).")
    parser.add_argument("--highlight_dir", type=str, required=True,
                        help="Directory of highlight videos (144p).")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to write per-sport CLIP-similarity JSON files.")
    parser.add_argument("--sports", nargs="+", default=SPORTS, choices=SPORTS,
                        help="Sports to process (default: all).")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    with open(args.filtered_json, "r", encoding="utf-8") as f:
        matching_results = json.load(f)

    for sport in args.sports:
        sport_results = {k: v for k, v in matching_results.items() if k.startswith(sport)}
        if not sport_results:
            continue
        output_path = os.path.join(args.output_dir, f"{sport}.json")
        if os.path.exists(output_path):
            print(f"{output_path} already exists, skipping.")
            continue

        metric = {}
        for key, frame_list in tqdm(sport_results.items(), desc=f"CLIP sim {sport}"):
            full_path = os.path.join(args.full_dir, f"{key}.mp4")
            hi_path = os.path.join(args.highlight_dir, f"{key}.mp4")
            if not (os.path.exists(full_path) and os.path.exists(hi_path)):
                print(f"[skip] missing video for {key}")
                continue

            full_frames_all = extract_all_frames(full_path)
            highlight_frames_all = extract_middle_sampled_frames(hi_path)

            full_frames, highlight_frames = [], []
            for i, idx in enumerate(frame_list):
                if idx is not None:
                    full_frames.append(full_frames_all[idx])
                    highlight_frames.append(highlight_frames_all[i])

            metric[key] = clip_sim(np.stack(full_frames), np.stack(highlight_frames))
            print(f"{key}: {metric[key]:.4f}")

        with open(output_path, 'w') as f:
            json.dump(metric, f, indent=4)
        print(f"Saved CLIP similarity to {output_path}")


if __name__ == "__main__":
    main()
