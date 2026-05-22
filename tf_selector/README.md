# SVHighlights — TF-SELECTOR preprocessing

Preprocessing code that extracts the inputs required by the **TF-SELECTOR**
baseline (paper Section 4). For reproducing the SVHighlights dataset itself see
[`../benchmark/`](../benchmark/).

TF-SELECTOR is a training-free pipeline; this directory covers only its
preprocessing stages. The captioning / scoring / evaluation stages are part of
the model code and are maintained separately.

## Pipeline

| Step | Script | Paper | Description |
|---|---|---|---|
| 1 | `shot_boundary.py` | §4.1 | Detect shot boundaries with TransNet V2. |
| 2 | `transcribe.py` | §4.1 | Word-level speech recognition with WhisperX. |
| 3 | `segment.py` | §4.1 | Merge shots + transcripts into context-aware segments. |
| 4 | `volume.py` | §4.3 | Per-clip (2-second) audio loudness in dBFS. |
| 5 | `volume_minmax.py` | §4.3 | Normalize the dBFS values to the [0, 1] range used by the scoring LLM. |

## Setup

`benchmark/` and `tf_selector/` share a single conda environment — see
[the root README](../README.md#environment-setup) for setup instructions,
including TransNet V2 (needed by `shot_boundary.py`) and WhisperX (needed by
`transcribe.py`).

## Data layout

See [the root README](../README.md#data-layout) for the directory layout used
by the preprocessing pipeline, and its [Dataset](../README.md#dataset) section
for the released annotations and features.

## Usage

```bash
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

Every script accepts `--sports` to restrict processing to a subset of sports,
e.g. `--sports soccer basketball`.

## Notes

- **`segment.py --max_duration`** defaults to `120` seconds — the 2-minute
  maximum segment length from §5.1.1. It is exposed as a CLI argument.
- **TransNet V2** is a third-party tool and is not bundled here; clone it
  (see Setup) and pass its inference script via `--transnetv2_script`.
