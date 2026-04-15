import os
import json
import time
import shutil
from tqdm import tqdm

def backup_single_prompt(scene_path):
    src_path = os.path.join(scene_path, "prompts.json")
    dst_dir = os.path.join(scene_path, "backup")
    dst_path = os.path.join(dst_dir, "prompts.json.bak")
    os.makedirs(dst_dir, exist_ok=True)

    if not os.path.exists(src_path) or os.path.getsize(src_path) < 1:
        return None
    
    data = {}
    if os.path.exists(dst_path) and os.path.getsize(dst_path) > 0:
        with open(dst_path, "r") as f:
            data = json.load(f)
    
    id = len(data.keys())
    with open(src_path, "r") as f:
        data[id] = json.load(f)
    
    with open(dst_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def backup_all_prompt():
    # root_dir = "/data1/cympyc1785/SceneData/DL3DV/scenes"
    # splits = ["1K", "2K", "3K", "4K", "5K", "6K", "7K"]

    root_dir = "/data1/cympyc1785/SceneData/DynamicVerse/scenes"
    splits = ["DAVIS", "MOSE", "MVS-Synth", "SAV", "VOST", "dynamic_replica", "spring", "youtube_vis", "uvo"]

    root_dir = "/data1/cympyc1785/SceneData/DynamicVerse/scenes/dynpose-100k"
    splits = [f"dynpose-{i:04d}" for i in range(0, 50)]

    for split in splits:
        data_dir = os.path.join(root_dir, split)

        print("Processing split", split)
        for h in tqdm(sorted(os.listdir(data_dir))):
            scene_path = os.path.join(data_dir, h)
            backup_single_prompt(scene_path)
    
if __name__ == "__main__":
    backup_all_prompt()