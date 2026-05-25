import os

import pandas as pd

from torch.utils.data import Dataset

from util import load_json

class ELVHD(Dataset):
    def __init__(self, args, mode):
        super().__init__()
        self.args = args
        self.mode = mode

        self.metadata = self.get_metadata()
        self.segment = self.get_data(self.args.segment_path) if os.path.exists(self.args.segment_path) else {}
        self.volume = self.get_data(self.args.volume_path) if os.path.exists(self.args.volume_path) else {}

        self.data = self.build_data()

    def get_metadata(self):
        metadata = pd.read_csv(self.args.meta_path)
        return metadata

    def get_data(self, data_path):
        data = load_json(data_path)
        return data
    
    def get_video_frame(self, frame_path):
        new_frame_path = []
        for frame in os.listdir(frame_path):
            new_frame_path.append(os.path.join(frame_path, frame))
        return new_frame_path
    
    def get_frame_per_segment(self, vid):
        frame_path = os.path.join(self.args.video_path, vid)
        frame_path = self.get_video_frame(frame_path)
        segment_info = self.segment[vid]
        curr_frame_idx = 0
        num_frames = len(frame_path)
        for item in segment_info:
            frame_per_segment = []
            start, end = float(item["start"]), float(item["end"])
            while curr_frame_idx < num_frames:
                if curr_frame_idx < start:
                    curr_frame_idx += 2
                    continue
                if curr_frame_idx > end:
                    break
                frame_per_segment.append(frame_path[curr_frame_idx])
                curr_frame_idx += 2 # 2s clip

            item["frame_path"] = frame_per_segment

        return segment_info

    def get_caption_volume_per_segment(self, vid):
        volume_info = self.volume[vid]
        segment_info = self.segment[vid]
        curr_frame_idx = 0
        num_frames = len(volume_info) * 2 - 1
        for item in segment_info:
            volume_per_segment = []
            start, end = float(item["start"]), float(item["end"])
            while curr_frame_idx < num_frames:
                if curr_frame_idx < start:
                    curr_frame_idx += 2
                    continue
                if curr_frame_idx > end:
                    break
                volume_per_segment.append(volume_info[curr_frame_idx//2])
                curr_frame_idx += 2 # 2s clip
            
            item["volume"] = volume_per_segment

        return segment_info

    def build_data(self):
        data = []
        for row in self.metadata.iterrows():
            if isinstance(row, tuple):
                row = row[-1]  # remove table index
            vid = row["vid"]
            full_duration = row["full_end"] - row["full_start"]
            if self.mode == "segment_captioning":
                segment_info = self.get_frame_per_segment(vid)
                for item in segment_info:
                    data.append({
                        "vid": vid,
                        "idx": str(item["idx"]),
                        "start": item["start"],
                        "end": item["end"],
                        "text": item["text"],
                        "segment_interval": item["segment_interval"],
                        "duration": item["duration"],
                        "full_duration": full_duration,
                        "frame_path": item["frame_path"],
                    })
            elif self.mode == "highlight_detection":
                segment_info = self.get_caption_volume_per_segment(vid)
                for item in segment_info:
                    data.append({
                        "vid": vid,
                        "idx": str(item["idx"]),
                        "start": item["start"],
                        "end": item["end"],
                        "text": item["text"],
                        "segment_interval": item["segment_interval"],
                        "duration": item["duration"],
                        "full_duration": full_duration,
                        "volume": item["volume"],
                        "caption": item["caption"]
                    })

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]