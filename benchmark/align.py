"""Highlight alignment (SVHighlights, Section 3.3).

For every 1-second clip of an official highlight video, this script finds the
most similar frame in the corresponding full-length broadcast.

  * Section 3.3.1 -- Finding the most similar frame:
        Videos are downsampled to 144p. We use *all* frames of the full video
        and the *middle* frame of each 1-second highlight clip, and match them
        with a pixel-level PSNR score (higher PSNR = more similar).

  * Section 3.3.2 -- Post-processing step:
        Temporal consistency is enforced. We default to the temporally
        expected frame f+ (one second after the previous match) and only
        switch to the global best match f* when its PSNR exceeds that of f+
        by more than a threshold tau (the paper adopts tau = 5).

Output: one JSON file per video pair under ``--output_dir`` named after the
full video, each a list of {"frame_idx", "psnr", "start", "end"} entries --
one per highlight clip.
"""
import argparse
import json
import os
import subprocess

import numpy as np
import torch
from natsort import natsorted
from tqdm import tqdm

SPORTS = ['american_football', 'baseball', 'basketball', 'ice_hockey',
          'race', 'rugby', 'soccer', 'volleyball']

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def extract_all_frames(video_path):
    """Decode every frame of a video into a list of BGR numpy arrays."""
    probe_cmd = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,nb_frames,avg_frame_rate,duration',
        '-of', 'json', video_path,
    ]
    probe = subprocess.run(probe_cmd, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, universal_newlines=True)
    info = json.loads(probe.stdout)
    stream = info['streams'][0]

    width = int(stream['width'])
    height = int(stream['height'])
    nb_frames = int(stream.get('nb_frames', 0))
    num, den = stream['avg_frame_rate'].split('/')
    fps = float(num) / float(den)
    duration = float(stream.get('duration', nb_frames / fps))

    command = [
        'ffmpeg', '-i', video_path,
        '-f', 'image2pipe', '-pix_fmt', 'bgr24',
        '-vcodec', 'rawvideo', '-',
    ]
    pipe = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    frames = []
    frame_size = width * height * 3  # bgr24: 3 bytes per pixel
    with tqdm(total=nb_frames if nb_frames > 0 else None,
              desc=f"Extracting all frames of {os.path.basename(video_path)}") as pbar:
        while True:
            raw_frame = pipe.stdout.read(frame_size)
            if len(raw_frame) != frame_size:
                break
            frame = np.frombuffer(raw_frame, np.uint8).reshape((height, width, 3))
            frames.append(frame)
            pbar.update(1)
    pipe.stdout.close()
    pipe.wait()
    return frames, fps, duration


def extract_middle_sampled_frames(video_path):
    """Decode the middle frame of every 1-second clip (0.5s, 1.5s, 2.5s, ...)."""
    probe_cmd = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,duration,avg_frame_rate',
        '-of', 'json', video_path,
    ]
    probe = subprocess.run(probe_cmd, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, universal_newlines=True)
    info = json.loads(probe.stdout)
    stream = info['streams'][0]

    width = int(stream['width'])
    height = int(stream['height'])
    duration = float(stream['duration'])
    num, den = stream['avg_frame_rate'].split('/')
    fps = float(num) / float(den)

    total_seconds = int(duration)
    frame_size = width * height * 3

    # -ss 0.5 + fps=1 yields one frame per second, sampled at clip midpoints.
    command = [
        'ffmpeg', '-ss', '0.5', '-i', video_path,
        '-vf', 'fps=1',
        '-f', 'image2pipe', '-pix_fmt', 'bgr24',
        '-vcodec', 'rawvideo', '-',
    ]
    pipe = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    frames = []
    with tqdm(total=total_seconds,
              desc=f"Extracting middle frames of {os.path.basename(video_path)}") as pbar:
        while True:
            raw_frame = pipe.stdout.read(frame_size)
            if len(raw_frame) != frame_size:
                break
            frame = np.frombuffer(raw_frame, np.uint8).reshape((height, width, 3))
            frames.append(frame)
            pbar.update(1)
    pipe.stdout.close()
    pipe.wait()
    return frames, fps, duration


def frames_to_vectors(frames):
    return np.stack(frames, axis=0)


def numpy_to_torch(vectors):
    return torch.tensor(vectors, dtype=torch.float32).to(DEVICE)


def calculate_psnr_batch(highlight_batch, full_batch, max_pixel_value=255.0):
    """PSNR for every (highlight, full) pair in the two broadcasted batches."""
    mse = torch.mean((highlight_batch - full_batch) ** 2, dim=(2, 3, 4))  # Bh x Bf
    psnr = 10 * torch.log10(max_pixel_value ** 2 / (mse + 1e-8))          # Bh x Bf
    return psnr


def find_matching_frames(full_video_vectors, highlight_vectors,
                         hi_batch_size, full_batch_size):
    """Section 3.3.1: for each highlight frame pick the full frame with max PSNR."""
    matching_frames = []
    psnr_scores = []

    for i in tqdm(range(0, len(highlight_vectors), hi_batch_size), desc="Matching frames"):
        highlight_batch = highlight_vectors[i:i + hi_batch_size]
        highlight_batch_tensor = numpy_to_torch(highlight_batch)  # Bh x H x W x C

        best_scores = torch.full((highlight_batch.shape[0],), float('-inf'), device=DEVICE)
        best_full_indices = torch.full((highlight_batch.shape[0],), -1,
                                       dtype=torch.long, device=DEVICE)

        for j in range(0, len(full_video_vectors), full_batch_size):
            full_batch = full_video_vectors[j:j + full_batch_size]  # Bf x H x W x C
            psnr_vals = calculate_psnr_batch(
                highlight_batch_tensor.unsqueeze(1),
                numpy_to_torch(full_batch).unsqueeze(0))           # Bh x Bf
            max_scores, max_indices = torch.max(psnr_vals, dim=1)  # Bh

            update_mask = max_scores > best_scores
            best_scores = torch.where(update_mask, max_scores, best_scores)
            best_full_indices = torch.where(update_mask, max_indices + j, best_full_indices)

        psnr_scores.extend(best_scores.cpu().tolist())
        matching_frames.extend(best_full_indices.cpu().tolist())

    return matching_frames, psnr_scores


def refine_matched_frames(full_video_vectors, highlight_vectors,
                          matching_frames, psnr_scores, fps, tau):
    """Section 3.3.2: enforce temporal consistency.

    The first highlight frame keeps its global best match. For every later
    frame we compare the global best match f* against the temporally expected
    frame f+ (one second after the previous aligned frame). We default to f+
    and only switch to f* when PSNR(f*) - PSNR(f+) > tau.
    """
    refined_frames = []
    refined_scores = []
    refine_details = []

    for idx in range(len(matching_frames)):
        match_idx = matching_frames[idx]
        orig_score = psnr_scores[idx]
        highlight_frame = highlight_vectors[idx]

        if idx == 0:
            refined_frames.append(match_idx)
            refined_scores.append(orig_score)
            refine_details.append({
                "highlight_idx": idx,
                "best_psnr_idx": match_idx,
                "best_psnr_score": orig_score,
                "candidate_1s_idx": None,
                "candidate_1s_score": None,
                "selected": "best_psnr",
            })
            continue

        # Temporally expected frame f+: one second after the previous match.
        prev_refined_idx = refined_frames[-1]
        candidate_idx = min(int(prev_refined_idx + fps), len(full_video_vectors) - 1)
        candidate_frame = full_video_vectors[candidate_idx]

        highlight_tensor = numpy_to_torch(highlight_frame).unsqueeze(0).unsqueeze(0)
        candidate_tensor = numpy_to_torch(candidate_frame).unsqueeze(0).unsqueeze(0)
        psnr_candidate = calculate_psnr_batch(highlight_tensor, candidate_tensor).squeeze().item()

        # Default to f+; switch to the global best match only if it is better
        # by more than tau.
        if abs(psnr_candidate - orig_score) <= tau:
            refined_frames.append(candidate_idx)
            refined_scores.append(psnr_candidate)
            selected = "candidate_1s"
        else:
            refined_frames.append(match_idx)
            refined_scores.append(orig_score)
            selected = "best_psnr"

        refine_details.append({
            "highlight_idx": idx,
            "best_psnr_idx": match_idx,
            "best_psnr_score": orig_score,
            "candidate_1s_idx": candidate_idx,
            "candidate_1s_score": psnr_candidate,
            "selected": selected,
        })

    return refined_frames, refined_scores, refine_details


def get_highlight_times(frame_indices, psnr_scores, fps, duration):
    """Convert aligned frame indices into 1-second highlight intervals.

    Each timestamp is turned into a 1-second interval centered on the frame
    (0.5s before and after), following Section 3.4.
    """
    highlight_times = []
    frame_time_interval = 1 / fps
    clip_duration = 0.5

    for frame_idx, psnr in zip(frame_indices, psnr_scores):
        frame_idx = int(frame_idx)
        frame_time = frame_idx * frame_time_interval
        start_time = max(0, frame_time - clip_duration)
        end_time = min(duration, frame_time + clip_duration)
        highlight_times.append({
            "frame_idx": frame_idx, "psnr": psnr,
            "start": start_time, "end": end_time,
        })
    return highlight_times


def process_pair(full_video_path, highlight_video_path, output_file_path, args):
    full_video_frames, fps, duration = extract_all_frames(full_video_path)
    highlight_frames, _, _ = extract_middle_sampled_frames(highlight_video_path)

    full_video_vectors = frames_to_vectors(full_video_frames)
    highlight_vectors = frames_to_vectors(highlight_frames)

    print("matching...")
    matching_frames, psnr_scores = find_matching_frames(
        full_video_vectors, highlight_vectors, args.hi_batch_size, args.full_batch_size)
    matching_frames, psnr_scores, refine_details = refine_matched_frames(
        full_video_vectors, highlight_vectors, matching_frames, psnr_scores, fps, args.tau)

    print("getting timestamps...")
    highlight_times = get_highlight_times(matching_frames, psnr_scores, fps, duration)

    with open(output_file_path, 'w') as f:
        json.dump(highlight_times, f, indent=4)
    print(f"Saved alignment result to {output_file_path}")

    if args.save_refine_details:
        os.makedirs(args.save_refine_details, exist_ok=True)
        name = os.path.splitext(os.path.basename(output_file_path))[0]
        detail_path = os.path.join(args.save_refine_details, f"{name}_refine_details.json")
        with open(detail_path, 'w') as f:
            json.dump(refine_details, f, indent=4)
        print(f"Saved post-processing details to {detail_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Align highlight clips to full-video frames via PSNR (Section 3.3).")
    parser.add_argument("--full_dir", type=str, required=True,
                        help="Directory of trimmed full-length videos (144p).")
    parser.add_argument("--highlight_dir", type=str, required=True,
                        help="Directory of highlight videos (144p).")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to write per-video alignment JSON files.")
    parser.add_argument("--sports", nargs="+", default=SPORTS, choices=SPORTS,
                        help="Sports to process (default: all).")
    parser.add_argument("--hi_batch_size", type=int, default=128,
                        help="Batch size for highlight frames.")
    parser.add_argument("--full_batch_size", type=int, default=128,
                        help="Batch size for full-video frames (lower it if GPU OOM).")
    parser.add_argument("--tau", type=float, default=5.0,
                        help="Post-processing PSNR-gap threshold (Section 3.3.2, paper uses 5).")
    parser.add_argument("--save_refine_details", type=str, default=None,
                        help="Optional directory to dump post-processing decision details.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for sport in args.sports:
        full_files = natsorted(
            os.path.join(args.full_dir, f) for f in os.listdir(args.full_dir)
            if f.startswith(sport) and f.endswith(".mp4"))
        highlight_files = natsorted(
            os.path.join(args.highlight_dir, f) for f in os.listdir(args.highlight_dir)
            if f.startswith(sport) and f.endswith(".mp4"))

        if len(full_files) != len(highlight_files):
            print(f"[warn] {sport}: full ({len(full_files)}) and highlight "
                  f"({len(highlight_files)}) video counts differ; skipping.")
            continue

        for full_video_path, highlight_video_path in zip(full_files, highlight_files):
            name = os.path.splitext(os.path.basename(full_video_path))[0]
            output_file_path = os.path.join(args.output_dir, f"{name}.json")
            if os.path.exists(output_file_path):
                print(f"{output_file_path} already exists, skipping.")
                continue
            process_pair(full_video_path, highlight_video_path, output_file_path, args)


if __name__ == "__main__":
    main()
