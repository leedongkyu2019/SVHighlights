"""Matching-quality evaluation (SVHighlights, Section 3.3, Table 3).

Evaluates how well the highlight-alignment pipeline matched highlight clips to
full-video frames. Two kinds of metrics are reported:

  * Remaining rate -- the proportion of highlight clips kept after filtering
    (i.e. not mapped to ``null``). Computed by default.

  * PSNR / SSIM    -- pixel-level similarity between each highlight frame and
    its aligned full-video frame. Enabled with ``--full_metrics`` (slow: it
    re-decodes every frame of every video).

Input : the filtered-frame-index JSON produced by filter_frames.py.
Output: one JSON per sport under ``--output_dir`` plus an aggregated
        ``summary.json``.
"""
import argparse
import json
import os
import subprocess

import numpy as np
from tqdm import tqdm

SPORTS = ['american_football', 'baseball', 'basketball', 'ice_hockey',
          'race', 'rugby', 'soccer', 'volleyball']


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


def compute_psnr_ssim(full_video_path, highlight_video_path, frame_indices):
    """Average PSNR and SSIM over every aligned (non-null) highlight clip."""
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    all_frames = extract_all_frames(full_video_path)
    middle_frames = extract_middle_sampled_frames(highlight_video_path)

    psnr_values, ssim_values = [], []
    for i, idx in enumerate(frame_indices):
        if idx is None:
            continue
        full_np = np.array(all_frames[idx])
        hi_np = np.array(middle_frames[i])
        psnr_values.append(peak_signal_noise_ratio(full_np, hi_np, data_range=255))
        ssim_values.append(structural_similarity(
            full_np, hi_np, data_range=255, channel_axis=2))

    avg_psnr = float(np.mean(psnr_values)) if psnr_values else None
    avg_ssim = float(np.mean(ssim_values)) if ssim_values else None
    return avg_psnr, avg_ssim


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate highlight-alignment matching quality (Table 3).")
    parser.add_argument("--filtered_json", type=str, required=True,
                        help="Filtered-frame-index JSON (filter_frames.py output).")
    parser.add_argument("--full_dir", type=str, required=True,
                        help="Directory of full-length videos (144p).")
    parser.add_argument("--highlight_dir", type=str, required=True,
                        help="Directory of highlight videos (144p).")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to write per-sport and summary metrics.")
    parser.add_argument("--sports", nargs="+", default=SPORTS, choices=SPORTS,
                        help="Sports to process (default: all).")
    parser.add_argument("--full_metrics", action="store_true",
                        help="Also compute PSNR/SSIM (slow: re-decodes every frame).")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    with open(args.filtered_json, "r", encoding="utf-8") as f:
        matching_results = json.load(f)

    summary = {}
    for sport in args.sports:
        sport_results = {k: v for k, v in matching_results.items() if k.startswith(sport)}
        if not sport_results:
            continue

        metrics = {}
        for key, frame_list in tqdm(sport_results.items(), desc=f"Evaluating {sport}"):
            entry = {
                "remaining_rate": len([f for f in frame_list if f is not None]) / len(frame_list)
            }
            if args.full_metrics:
                full_path = os.path.join(args.full_dir, f"{key}.mp4")
                hi_path = os.path.join(args.highlight_dir, f"{key}.mp4")
                if os.path.exists(full_path) and os.path.exists(hi_path):
                    psnr, ssim = compute_psnr_ssim(full_path, hi_path, frame_list)
                    entry["psnr"], entry["ssim"] = psnr, ssim
            metrics[key] = entry

        with open(os.path.join(args.output_dir, f"{sport}.json"), "w") as f:
            json.dump(metrics, f, indent=4)

        rates = [m["remaining_rate"] for m in metrics.values()]
        summary[sport] = {"remaining_rate": round(float(np.mean(rates)) * 100, 2)}
        if args.full_metrics:
            psnrs = [m["psnr"] for m in metrics.values() if m.get("psnr") is not None]
            ssims = [m["ssim"] for m in metrics.values() if m.get("ssim") is not None]
            if psnrs:
                summary[sport]["psnr"] = round(float(np.mean(psnrs)), 2)
                summary[sport]["ssim"] = round(float(np.mean(ssims)), 3)

    if summary:
        overall = round(float(np.mean([s["remaining_rate"] for s in summary.values()])), 2)
        summary["all"] = {"remaining_rate": overall}
        print(f"Overall remaining rate: {overall}%")

    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=4)
    print(f"Saved summary to {os.path.join(args.output_dir, 'summary.json')}")


if __name__ == "__main__":
    main()
