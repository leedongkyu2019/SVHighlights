import re
import os
import argparse

from util import *

def parse_output(output):
    pattern = r"The segment timestamps are (\d+\.\d{3})s ~ (\d+\.\d{3})s\. Its saliency score is (\d+\.\d*)"
    match = re.match(pattern, output)
    if match:
        start = float(match.group(1))
        end = float(match.group(2))
        score = float(match.group(3))
        return start, end, score
    else:
        return None, None, None


def format_vhd_output(pred):
    parsed_pred = []
    cnt, cnt2 = 0, 0
    for i, vid in enumerate(pred):
        video_type = "_".join(vid.split("_")[:-1])
        query = f"Highlight of this {video_type} video."
        full_length = len(os.listdir(f"path/to/frame/{vid}"))
        clip_scores = [0 for _ in range(0, full_length, 2)]
        outputs = pred[vid]
        for idx, _ in enumerate(outputs):
            pred_saliency_score = outputs[idx]["saliency_score"]
            segment_start, segment_end, score = parse_output(pred_saliency_score)
            # print(vid, len(segment[vid]), idx)
            r_start, r_end = float(outputs[idx]["start"]), float(outputs[idx]["end"])
            if r_start != segment_start or r_end != segment_end:
                # print('err')
                cnt2 += 1
                continue
            if score is None:
                # print(vid, j, outputs[idx])
                cnt += 1
                continue
            if score > 5.0:
                score = 5.0
            elif score < 0.0:
                score = 0.0
            # assign score to clip id between start~end (overlap as weight * score)
            for k in range(len(clip_scores)):
                clip_start, clip_end = k * 2.0, (k+1) * 2.0
                overlap_start = max(segment_start, clip_start)
                overlap_end = min(segment_end, clip_end)
                if overlap_start < overlap_end:
                    weight = (overlap_end - overlap_start) / 2.0
                    # print(clip_start, clip_end, segment_start, segment_end, weight, score, clip_scores[k])
                    clip_scores[k] += weight * score
                    # print(clip_scores[k])
        clip_scores = [round(score, 1) for score in clip_scores]
        result = {}
        result["query"] = query
        result["vid"] = vid
        result["pred_saliency_scores"] = clip_scores
        parsed_pred.append(result)
    #     print(vid)
    
    # print("parsing error", cnt)
    # print(cnt2)
    
    return parsed_pred

def main():
    parser = argparse.ArgumentParser("")
    parser.add_argument("--pred_path", type=str)
    parser.add_argument("--save_path", type=str)

    args = parser.parse_args()

    pred = load_json(args.pred_path)

    parsing_pred = format_vhd_output(pred)

    save_json(parsing_pred, args.save_path, None)

if __name__ == "__main__":
    main()