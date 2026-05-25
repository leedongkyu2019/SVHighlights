import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from util import *
from model.vlm import get_model
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

    # run (captioning)
    output_data = {}
    pbar = tqdm(total=len(dataset))
    for i, item in enumerate(dataset):            
        prompt = f"Please describe this segment."
        prompts = [[prompt, item["frame_path"]]]
        if len(item["frame_path"]) != 0:
            preds, infos = model.forward(None, prompts)
        else:
            preds = [""]
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
            # "frame_path": item["frame_path"],
            "caption": preds[0]
        })
        if i % args.save_every == 0:
            save_json(output_data, output_path)
        pbar.update(1)

    save_json(output_data, output_path)

if __name__ == '__main__':
    launch()
    