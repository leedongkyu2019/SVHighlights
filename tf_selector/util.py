import json
import argparse

def load_jsonl(filename):
    with open(filename, "r") as f:
        return [json.loads(l.strip("\n")) for l in f.readlines()]

def load_json(data_path):
    with open(data_path, "r") as f:
        data = json.load(f)
    return data

def save_json(data, data_path, indent=4):
    with open(data_path, "w") as f:
        json.dump(data, f, indent=indent)

def read_txt(data_path):
    with open(data_path, "r") as fin:
        data = fin.readline().strip()
    return data

def parse_args():
    parser = argparse.ArgumentParser("")

    # data
    parser.add_argument("--meta_path", default="", type=str) 
    parser.add_argument("--video_path", default="", type=str)
    parser.add_argument("--segment_path", default="", type=str) 
    parser.add_argument("--volume_path", default="", type=str)
    parser.add_argument("--batch_size", default=1, type=int)

    # mode
    parser.add_argument("--mode", choices=["segment_captioning", "highlight_detection"])

    # output
    parser.add_argument("--output_path", required=True, type=str)  
    parser.add_argument("--output_filename", required=True, type=str)  

    # model
    parser.add_argument("--model", type=str)
    parser.add_argument("--temperature", default=0.0, type=float)
    parser.add_argument("--max_new_tokens", default=1024, type=int)  

    # other
    parser.add_argument("--start_from_scratch", action='store_true')
    parser.add_argument("--save_every", default=1000, type=int)

    return parser.parse_args()