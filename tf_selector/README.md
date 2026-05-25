# SVHighlights — TF-SELECTOR

End-to-end code for the **TF-SELECTOR** baseline (paper Section 4):
preprocessing (shot boundaries, ASR, context-aware segmentation, audio volume)
followed by training-free inference (VLM segment captioning, LLM saliency
scoring, and prediction parsing). For reproducing the SVHighlights dataset
itself see [`../benchmark/`](../benchmark/).

## Pipeline

> **All preprocessing and captioning outputs (steps 1–6) are released with the
> dataset** under `annotations/`, so steps 1–6 are **optional** — you only need
> to re-run them to regenerate the artifacts yourself. To evaluate TF-SELECTOR
> with the released artifacts, jump straight to steps 7–8.

Preprocessing (cwd-independent — call from anywhere) — **all optional**:

| Step | Script | Paper | Description |
|---|---|---|---|
| 1 | `shot_boundary.py` | §4.1 | Detect shot boundaries with TransNet V2. **Optional** — released as `annotations/shots/`. |
| 2 | `transcribe.py` | §4.1 | Word-level speech recognition with WhisperX. **Optional** — released as `annotations/whisper/`. |
| 3 | `segment.py` | §4.1 | Merge shots + transcripts into context-aware segments. **Optional** — released as `annotations/segments/`. |
| 4 | `volume.py` | §4.3 | Per-clip (2-second) audio loudness in dBFS. **Optional** — released as `annotations/volume.json`. |
| 5 | `volume_minmax.py` | §4.3 | Normalize the dBFS values to the [0, 1] range used by the scoring LLM. **Optional** — released as `annotations/minmax_volume.json`. |

Inference (run from inside `tf_selector/` — the scripts import sibling modules `util`, `dataset`, `model.*`):

| Step | Script | Paper | Description |
|---|---|---|---|
| 6 | `segment_captioning.py` | §4.2 | Segment-level captioning with a VLM (InternVL2_5-8B). **Optional** — released as `annotations/segment_caption.json`. |
| 7 | `main.py` | §4.3 | LLM-based per-segment saliency scoring (Llama-3-8B-Instruct). |
| 8 | `parse.py` | §4.3 | Parse the LLM output into the per-clip saliency-score JSON consumed by `eval.py`. |

## Usage

```bash
# Steps 1–5 are OPTIONAL — every output is already released under annotations/
# in the Hugging Face dataset. Run them only to regenerate the artifacts.

# 1. Shot boundary detection (§4.1)
python shot_boundary.py \
    --video_dir data/videos/full/144p \
    --output_dir data/annotations/shots \
    --transnetv2_script /path/to/TransNetV2/inference/transnetv2.py

# 2. Speech recognition (§4.1)
python transcribe.py \
    --video_dir data/videos/full/144p \
    --whisper_dir data/annotations/whisper \
    --whisper_all_dir data/annotations/whisper_all

# 3. Context-aware segmentation (§4.1)
python segment.py \
    --whisper_dir data/annotations/whisper \
    --video_dir data/videos/full/144p \
    --shot_dir data/annotations/shots \
    --output_dir data/annotations/segments

# 4. Audio volume extraction (§4.3)
python volume.py \
    --video_dir data/videos/full/144p \
    --output_dir data/annotations/volume

# 5. Volume normalization (§4.3)
python volume_minmax.py \
    --volume_dir data/annotations/volume \
    --output data/annotations/volume_norm.json
```

Every preprocessing script accepts `--sports` to restrict processing to a
subset of sports, e.g. `--sports soccer basketball`.

```bash
# Inference — run from inside tf_selector/ (sibling-module imports).
cd tf_selector

# 6. (Optional) Segment-level captioning (§4.2)
#    We release the VLM output as annotations/segment_caption.json; skip this
#    step and pass the released file to main.py if you do not need to recompute.
python segment_captioning.py \
    --meta_path ../data/metadata/video_list.csv \
    --video_path path/to/frames \
    --segment_path ../data/annotations/segments.json \
    --mode segment_captioning \
    --output_path output --output_filename segment_caption.json \
    --model OpenGVLab/InternVL2_5-8B --save_every 10

# 7. LLM-based per-segment saliency scoring (§4.3)
#    (use ../data/annotations/segment_caption.json, or output/segment_caption.json from step 6)
python main.py \
    --meta_path ../data/metadata/video_list.csv \
    --volume_path ../data/annotations/minmax_volume.json \
    --segment_path ../data/annotations/segment_caption.json \
    --mode highlight_detection \
    --output_path output --output_filename pred.json \
    --model meta-llama/Meta-Llama-3-8B-Instruct --save_every 10

# 8. Parse the LLM output into the per-clip saliency-score format eval.py expects
python parse.py --pred_path output/pred.json --save_path output/predictions.json
```

## Notes

- **`segment.py --max_duration`** defaults to `120` seconds — the 2-minute
  maximum segment length from §5.1.1. It is exposed as a CLI argument.
- **TransNet V2** is a third-party tool and is not bundled here; clone it
  (see Setup) and pass its inference script via `--transnetv2_script`.
- **Hugging Face token** — `main.py` and `segment_captioning.py` log in to
  the Hub at startup (`login(token='your_huggingface_token')`); replace the
  placeholder with your own token before running.
- **Frame directory** — `parse.py` reads the per-video frame directory at
  `path/to/frame/{vid}` (used to determine the video length); replace the
  placeholder with the directory that holds your per-video frames.
