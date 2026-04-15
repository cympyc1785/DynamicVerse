from collections import defaultdict
import random
from concurrent.futures import ProcessPoolExecutor, as_completed

def get_tasks(root_dir, splits):
    tasks = []
    for split in splits:
        split_dir = os.path.join(root_dir, split)
        for scene_name in sorted(os.listdir(split_dir)):
            scene_dir = os.path.join(split_dir, scene_name)
            tasks.append((scene_dir, split, scene_name))
    return tasks

def get_seg_stat(tasks):
    total_caption_cnt = 0
    for task in tasks:
        scene_dir, split, scene_name = task
        images_dir = os.path.join(scene_dir, "rgb")
        if not os.path.exists(images_dir):
            images_dir = os.path.join(scene_dir, "images")
        if not os.path.exists(images_dir):
            print("no img", images_dir)
            continue
        caption_path = os.path.join(scene_dir, "prompts.json")
        if not os.path.exists(caption_path):
            print("no cap", images_dir)
            continue
        with open(caption_path, "r") as f:
            data = json.load(f)
        seg_len = len(data.keys())
        total_caption_cnt += seg_len
    return total_caption_cnt

def get_all_tasks():
    root_dir = "/data1/cympyc1785/SceneData/DynamicVerse/scenes"
    splits = ["DAVIS", "MOSE", "MVS-Synth", "SAV", "VOST", "dynamic_replica", "spring", "uvo", "youtube_vis"]
    tasks = get_tasks(root_dir, splits)

    root_dir = "/data1/cympyc1785/SceneData/DynamicVerse/scenes/dynpose-100k"
    splits = [f"dynpose-{i:04d}" for i in range(0, 90)]
    tasks += get_tasks(root_dir, splits)

    root_dir = "/data1/cympyc1785/SceneData/DL3DV/scenes"
    splits = [f"{i}K" for i in range(1, 8)]
    tasks += get_tasks(root_dir, splits)

    return tasks

def get_process_time(sample_num=100):
    """
    For estimating total processing time
    """

    tasks = get_all_tasks()
    seg_num = get_seg_stat(tasks)
    print("Segment Count:", seg_num)

    tasks = random.sample(tasks, sample_num)

    t = time.time()
    with ProcessPoolExecutor(max_workers=4) as ex:
        futures = [ex.submit(worker, t) for t in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            fut.result()
    avg_process_time = (time.time() - t) / sample_num

    estimated_process_time = avg_process_time * seg_num
    print(estimated_process_time, "sec")
    print(estimated_process_time/3600, "hours")
    print(estimated_process_time/3600/24, "days")

    exit()