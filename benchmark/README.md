# SVHighlights — Benchmark (dataset construction)

Code for reproducing the **SVHighlights** dataset from the source videos
(paper Section 3). For the TF-SELECTOR baseline's preprocessing see
[`../tf_selector/`](../tf_selector/).

## Pipeline

| Step | Script | Paper | Description |
|---|---|---|---|
| 1 | `trim_video.py` | §3.2 | Trim full-length broadcasts down to the actual game footage. |
| 2 | `align.py` | §3.3.1–3.3.2 | Align each highlight clip to a full-video frame via PSNR, then enforce temporal consistency. |
| 3 | `filter_frames.py` | §3.3.3 | Automatically remove aligned frames with PSNR below a threshold. |
| 4 | `labeling.py` | §3.4 | Convert the aligned frames into per-clip (2-second) 0/1 highlight labels. |

### Evaluation

| Script | Paper | Description |
|---|---|---|
| `eval_matching_quality.py` | Table 2 | Remaining rate (and optionally PSNR/SSIM) of the alignment. |
| `eval_clip_similarity.py` | Table 2 | CLIP similarity between aligned frame pairs. |

> **Not included:** video download (the dataset is released as video URLs only;
> downloading from YouTube is the user's responsibility) and frame sampling
> (not needed for dataset construction — `labeling.py` derives the clip count
> directly from the video duration).

## Setup

`benchmark/` and `tf_selector/` share a single conda environment — see
[the root README](../README.md#environment-setup) for setup instructions.
`eval_clip_similarity.py` additionally needs the CLIP package (also covered
there).

## Data layout

See [the root README](../README.md#data-layout) for the directory layout used
by the preprocessing pipeline, and its [Dataset](../README.md#dataset) section
for the released annotations and features.

## Usage

```bash
# 1. Trim full videos to game footage (§3.2)
python trim_video.py \
    --video_list data/metadata/video_list.csv \
    --src_dir data/videos/full_raw \
    --dst_dir data/videos/full/144p

# 2. Align highlight clips to full-video frames (§3.3.1–3.3.2)
python align.py \
    --full_dir data/videos/full/144p \
    --highlight_dir data/videos/highlight/144p \
    --output_dir data/annotations/alignment

# 3. Automatic PSNR filtering (§3.3.3)
python filter_frames.py \
    --alignment_dir data/annotations/alignment \
    --output data/annotations/all_filtered_frame_idx.json

# 4. Generate per-clip highlight labels (§3.4)
python labeling.py \
    --filtered_json data/annotations/all_filtered_frame_idx.json \
    --video_dir data/videos/full/144p \
    --output data/annotations/label.json
```

### Evaluation

```bash
# Matching quality — remaining rate (Table 2). Add --full_metrics for PSNR/SSIM.
python eval_matching_quality.py \
    --filtered_json data/annotations/all_filtered_frame_idx.json \
    --full_dir data/videos/full/144p \
    --highlight_dir data/videos/highlight/144p \
    --output_dir data/eval/matching_quality

# CLIP similarity (Table 2)
python eval_clip_similarity.py \
    --filtered_json data/annotations/all_filtered_frame_idx.json \
    --full_dir data/videos/full/144p \
    --highlight_dir data/videos/highlight/144p \
    --output_dir data/eval/clip_similarity
```

Every script accepts `--sports` to restrict processing to a subset of sports,
e.g. `--sports soccer basketball`.

## Notes

- **Default parameters follow the paper.** `align.py --tau` defaults to `5`
  (the post-processing threshold τ adopted in Table 4). `filter_frames.py
  --psnr_threshold` defaults to `20`, and `labeling.py` uses 2-second clips
  with a 50% overlap threshold (§3.4). All are exposed as CLI arguments.
- **Manual filtering.** Section 3.3.3 applies a lightweight manual refinement
  on top of `filter_frames.py`. That step is a human visual check and is not
  scripted here.
- **Memory.** `align.py` decodes every frame of a full video into memory.
  Even at 144p an hours-long broadcast is large — run it on a high-RAM machine
  and lower `--full_batch_size` if the GPU runs out of memory.
