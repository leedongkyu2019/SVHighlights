"""
SVHighlights highlight-detection evaluation.

Given a model's per-clip saliency predictions and the dataset's ground-truth
highlight labels, reports four metrics per sport and over `all`:

    HL-mAP   clip-level mean Average Precision
    HL-Hit1  whether the top-scored clip is a ground-truth highlight
    HL-Hitk  hit rate among the top-K clips (K = #GT highlight clips)
    HL-IoU   temporal overlap between the top-K predictions and GT highlights

Usage:
    python eval.py --submission_path preds.json --gt_path gt.json --save_path out.json
"""
import json
import os

import numpy as np
import multiprocessing as mp

from collections import OrderedDict
from sklearn.metrics import precision_recall_curve


def load_json(data_path):
    with open(data_path, "r") as f:
        return json.load(f)


def compute_hl_hit1(vid2preds, vid2gt_scores):
    vid2max_scored_clip_idx = {k: np.argmax(v) for k, v in vid2preds.items()}
    hit_scores = np.zeros((len(vid2preds)))
    vids = list(vid2preds.keys())
    for idx, vid in enumerate(vids):
        pred_clip_idx = vid2max_scored_clip_idx[vid]
        gt_scores = vid2gt_scores[vid]
        if pred_clip_idx < len(gt_scores):
            hit_scores[idx] = gt_scores[pred_clip_idx]
    hit_at_one = float(f"{100 * np.mean(hit_scores):.2f}")
    return hit_at_one


def compute_hl_ap(vid2preds, vid2gt_scores, num_workers=8, chunksize=50):
    vid2pred_scores = {k: v for k, v in vid2preds.items()}
    ap_scores = np.zeros((len(vid2preds)))
    vids = list(vid2preds.keys())
    input_tuples = []
    for idx, vid in enumerate(vids):
        y_true = vid2gt_scores[vid]
        y_predict = np.array(vid2pred_scores[vid])
        input_tuples.append((idx, y_true, y_predict))

    if num_workers > 1:
        with mp.Pool(num_workers) as pool:
            for idx, score in pool.imap_unordered(
                    compute_ap_from_tuple, input_tuples, chunksize=chunksize):
                ap_scores[idx] = score
    else:
        for input_tuple in input_tuples:
            idx, score = compute_ap_from_tuple(input_tuple)
            ap_scores[idx] = score

    mean_ap = float(f"{100 * np.mean(ap_scores):.2f}")
    return mean_ap


def compute_ap_from_tuple(input_tuple):
    idx, y_true, y_predict = input_tuple
    if len(y_true) < len(y_predict):
        y_predict = y_predict[:len(y_true)]
    elif len(y_true) > len(y_predict):
        _y_predict = np.zeros(len(y_true))
        _y_predict[:len(y_predict)] = y_predict
        y_predict = _y_predict

    score = get_ap(y_true, y_predict)
    return idx, score


def get_ap(y_true, y_predict, interpolate=True, point_11=False):
    """
    Average precision in different formats: (non-) interpolated and/or 11-point approximated
    point_11=True and interpolate=True corresponds to the 11-point interpolated AP used in
    the PASCAL VOC challenge up to the 2008 edition and has been verfied against the vlfeat implementation
    The exact average precision (interpolate=False, point_11=False) corresponds to the one of vl_feat

    :param y_true: list/ numpy vector of true labels in {0,1} for each element
    :param y_predict: predicted score for each element
    :param interpolate: Use interpolation?
    :param point_11: Use 11-point approximation to average precision?
    :return: average precision

    ref: https://github.com/gyglim/video2gif_dataset/blob/master/v2g_evaluation/__init__.py

    """
    # Check inputs
    assert len(y_true) == len(y_predict), "Prediction and ground truth need to be of the same length"
    if len(set(y_true)) == 1:
        if y_true[0] == 0:
            return 0  # True labels are all zeros
        else:
            return 1
    else:
        assert sorted(set(y_true)) == [0, 1], "Ground truth can only contain elements {0,1}"

    # Compute precision and recall
    precision, recall, _ = precision_recall_curve(y_true, y_predict)
    recall = recall.astype(np.float32)

    if interpolate:  # Compute the interpolated precision
        for i in range(1, len(precision)):
            precision[i] = max(precision[i - 1], precision[i])

    if point_11:  # Compute the 11-point approximated AP
        precision_11 = [precision[np.where(recall >= t)[0][-1]] for t in np.arange(0, 1.01, 0.1)]
        return np.mean(precision_11)
    else:  # Compute the AP using precision at every additionally recalled sample
        indices = np.where(np.diff(recall))
        return np.mean(precision[indices])


def compute_hl_hitk(vid2preds, vid2gt_scores):
    hitk = 0.0
    for vid in vid2preds:
        scores = vid2preds[vid]
        gt_scores = vid2gt_scores[vid]

        indices_of_hl = [i for i, val in enumerate(gt_scores) if val == 1]
        k = len(indices_of_hl)

        if k == 0:
            continue

        topk_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

        correct = sum(1 for idx in topk_indices if idx in indices_of_hl)
        hitk += correct / k

    hitk /= len(vid2preds)

    return round(hitk * 100, 2)


def compute_hl_iou(vid2preds, vid2gt_scores):
    total_iou = 0.0

    for vid in vid2preds:
        scores = vid2preds[vid]
        gt_scores = vid2gt_scores[vid]

        min_len = min(len(scores), len(gt_scores))

        scores = scores[:min_len]
        gt_scores = gt_scores[:min_len]

        indices_of_hl = [i for i, val in enumerate(gt_scores) if val == 1]
        k = len(indices_of_hl)

        if k == 0:
            continue

        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        threshold = scores[sorted_indices[k - 1]]
        topk_indices = [i for i in sorted_indices if scores[i] >= threshold]

        pred_mask = [1 if i in topk_indices else 0 for i in range(len(scores))]
        gt_mask = [1 if val == 1 else 0 for val in gt_scores]

        intersection = sum(1 for i in range(len(scores)) if pred_mask[i] == 1 and gt_mask[i] == 1)
        union = sum(1 for i in range(len(scores)) if pred_mask[i] == 1 or gt_mask[i] == 1)
        iou = intersection / union if union > 0 else 0.0

        total_iou += iou

    total_iou /= len(vid2preds)

    return round(total_iou * 100, 2)


def eval_highlight(submission, ground_truth, verbose=True):
    highlight_det_metrics = {}
    video_domain = ["american_football", "baseball", "basketball", "ice_hockey",
                    "race", "rugby", "soccer", "volleyball", "all"]
    for domain in video_domain:
        vid2preds = {d["vid"]: d["pred_saliency_scores"] for d in submission if domain in d["vid"]} \
            if domain != "all" else {d["vid"]: d["pred_saliency_scores"] for d in submission}
        vid2gt_scores = {d["vid"]: d["saliency_scores"] for d in ground_truth if domain in d["vid"]} \
            if domain != "all" else {d["vid"]: d["saliency_scores"] for d in ground_truth}

        hit_at_one = compute_hl_hit1(vid2preds, vid2gt_scores)
        mean_ap = compute_hl_ap(vid2preds, vid2gt_scores)
        hit_at_k = compute_hl_hitk(vid2preds, vid2gt_scores)
        iou = compute_hl_iou(vid2preds, vid2gt_scores)
        highlight_det_metrics[domain] = {
            "HL-mAP": mean_ap, "HL-Hit1": hit_at_one, "HL-Hitk": hit_at_k,
            "HL-IoU": iou,
        }
    return highlight_det_metrics


def eval_submission(submission, ground_truth, verbose=True, match_number=False):
    """
    Args:
        submission: list(dict), each dict is {
            vid: str,
            pred_saliency_scores: list(float), len == #clips in video.
                i.e., each clip in the video will have a saliency score.
        }
        ground_truth: list(dict), each dict is {
          "vid": "baseball_1",
          "saliency_scores": []
               each element corresponds to one clip, in {0, 1}.
        }
    """
    pred_vids = set([e["vid"] for e in submission])
    gt_vids = set([e["vid"] for e in ground_truth])

    if match_number:
        assert pred_vids == gt_vids, \
            f"vids in ground_truth and submission must match. " \
            f"use `match_number=False` if you wish to disable this check"
    else:  # only leave the items that exists in both submission and ground_truth
        shared_vids = pred_vids.intersection(gt_vids)
        submission = [e for e in submission if e["vid"] in shared_vids]
        ground_truth = [e for e in ground_truth if e["vid"] in shared_vids]

    eval_metrics = {}
    eval_metrics_brief = OrderedDict()
    if "pred_saliency_scores" in submission[0]:
        highlight_det_scores = eval_highlight(submission, ground_truth, verbose=verbose)
        eval_metrics.update(highlight_det_scores)
        highlight_det_scores_brief = dict([
            (f"{'-'.join(k.split('-')[1:])}", v)
            for domain, _ in highlight_det_scores.items()
            for k, v in highlight_det_scores[domain].items()])
        eval_metrics_brief.update(highlight_det_scores_brief)

    # sort by keys
    final_eval_metrics = OrderedDict()
    final_eval_metrics["brief"] = eval_metrics_brief
    final_eval_metrics.update(sorted([(k, v) for k, v in eval_metrics.items()], key=lambda x: x[0]))
    return final_eval_metrics


def eval_main():
    import argparse
    parser = argparse.ArgumentParser(description="SVHighlights highlight-detection evaluation")
    parser.add_argument("--submission_path", type=str, required=True,
                        help="path to the prediction file (list of {vid, pred_saliency_scores})")
    parser.add_argument("--gt_path", type=str, required=True,
                        help="path to the ground-truth file (list of {vid, saliency_scores})")
    parser.add_argument("--save_path", type=str, required=True,
                        help="path to save the results JSON")
    parser.add_argument("--not_verbose", action="store_true")
    args = parser.parse_args()

    verbose = not args.not_verbose
    submission = load_json(args.submission_path)
    gt = load_json(args.gt_path)
    results = eval_submission(submission, gt, verbose=verbose)
    if verbose:
        print(json.dumps(results, indent=4))

    os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)
    with open(args.save_path, "w") as f:
        f.write(json.dumps(results, indent=4))


if __name__ == '__main__':
    eval_main()
