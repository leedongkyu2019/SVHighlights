import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from util import *
from model.llm import get_model
from dataset import ELVHD

from tqdm import tqdm
from huggingface_hub import login
login(token=os.environ.get('HF_TOKEN'))


def launch():
    # args
    args = parse_args()

    # output
    os.makedirs(args.output_path, exist_ok=True)
    output_path = os.path.join(args.output_path, args.output_filename)

    # resume
    # processed = {}
    # if not args.start_from_scratch and os.path.exists(output_path):
    #     processed = load_json(output_path)
    #     if 'data' in processed:
    #         processed = processed['data']

    # get dataset
    dataset = ELVHD(args, args.mode)

    # get model
    model = get_model(args)

    # run (highlight detection)
    output_data = {}
    pbar = tqdm(total=len(dataset))
    for i, item in enumerate(dataset):
        prompt = f"This is a segment corresponding to {item['start']}s ~ {item['end']}s from a video with a total length of {item['full_duration']:.3f} seconds. You are given the captions, volumes, and transcript for this segment. The volume values range from 0 to 1, where 1 indicates loud and 0 means quiet or silent. Please assign a saliency score to this segment on a scale from 0 to 5, where a higher score means the segment is more likely to be a highlight. The output format should be like: 'The segment timestamps are x.xxxs ~ x.xxxs. Its saliency score is x.x.'. You must not provide any other response or explanation, and the saliency score must be between 0 to 5.\nCaptions: {item['caption']}\nVolumes: {item['volume']}\nTranscript: {item['text']}"

        if item["caption"] != "":
            preds, infos = model.forward(None, prompt)
        else:
            preds = ""
        
        vid = item["vid"]
        if vid not in output_data:
            output_data[vid] = []

        output_data[vid].append({
            "idx": item["idx"],
            "start": item["start"],
            "end": item["end"],
            "text": item["text"],
            "segment_interval": item["segment_interval"],
            "duration": item["duration"],
            "full_duration": item["full_duration"],
            "volume": item["volume"],
            "caption": item["caption"],
            "saliency_score": preds if preds != "" else f"The segment timestamps are {item['start']}s ~ {item['end']}s. Its saliency score is 0.0."
        })
        if i % args.save_every == 0:
            save_json(output_data, output_path)
        pbar.update(1)
    
    save_json(output_data, output_path)

if __name__ == '__main__':
    launch()
    