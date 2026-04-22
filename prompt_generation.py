#!/usr/bin/env python3
"""
Use previously extracted keyframes to call QVQ-max API for dynamic object analysis
Based on high motion frame sampling results, not fixed frame count sampling
"""

from openai import OpenAI
import os
import base64
import json
import math
import glob
import argparse
import time
import numpy as np
from PIL import Image
from datetime import datetime
import shutil
from tqdm import tqdm
import imageio
import signal
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED


from utils.qwen_model_handler import QwenModelHandler

def encode_image(image_path):
    """Encode image file to base64 format"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def load_keyframe_paths(keyframes_dir):
    # Find PNG image files
    image_patterns = [
        os.path.join(keyframes_dir, "*.png"),
        os.path.join(keyframes_dir, "*.jpg"),
        os.path.join(keyframes_dir, "*.jpeg")
    ]
    
    image_paths = []
    for pattern in image_patterns:
        image_paths.extend(glob.glob(pattern))
    
    # Sort by filename
    image_paths.sort()

    return image_paths

def load_keyframes(image_paths, frame_interval=1, max_frames=64):
    if len(image_paths) > max_frames:
        image_paths = image_paths[:max_frames]
    
    # Load images
    frames = []
    frame_info = []
    
    for i in range(0, len(image_paths), frame_interval):
        image_path = image_paths[i]
        try:
            # Load image
            img = Image.open(image_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            frames.append(img)
            
            # Extract frame info (from filename)
            filename = os.path.basename(image_path)
            frame_info.append({
                'index': i,
                'filename': filename,
                'path': image_path
            })
            
        except Exception as e:
            print(f"Warning: Unable to load image {image_path}: {e}")
            continue
    
    if not frames:
        raise ValueError("Did not successfully load any images")
    
    return frames, frame_info

def find_image_folders(root_dir):
        """Find all scene folders in root directory, these folders should contain a subfolder named 'images'"""
        image_folders = []
        # print(f"Searching for scenes with 'images' subfolder in: {root_dir}")
        if not os.path.isdir(root_dir):
            print(f"Error: Root directory not found: {root_dir}")
            return []
        for scene_name in os.listdir(root_dir):
            scene_path = os.path.join(root_dir, scene_name)
            if os.path.isdir(scene_path):
                images_folder = os.path.join(scene_path, 'images')
                if os.path.isdir(images_folder):
                    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
                    try:
                        has_images = any(f.lower().endswith(tuple(image_extensions)) for f in os.listdir(images_folder))
                        if has_images:
                            image_folders.append(images_folder)
                            # print(f"Found image folder for scene '{scene_name}': {images_folder}")
                    except OSError as e:
                         print(f"Warning: Cannot read directory {images_folder}: {e}")
        # print(f"Total found {len(image_folders)} image folders")
        return sorted(image_folders)

def create_image_grid(images, num_columns=8):
    """
    Create image grid
    
    Args:
        images: List of PIL Image objects
        num_columns: Number of grid columns
    
    Returns:
        grid_image: Grid image
    """
    if not images:
        raise ValueError("Image list is empty")
    
    num_rows = math.ceil(len(images) / num_columns)
    
    # Get image size (assume all images have same size)
    img_width, img_height = images[0].size
    
    # If image sizes vary significantly, resize to appropriate size
    target_size = (min(img_width, 512), min(img_height, 384))  # Limit single image size
    
    resized_images = []
    for img in images:
        if img.size != target_size:
            img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
        else:
            img_resized = img
        resized_images.append(img_resized)
    
    # Create grid
    grid_width = num_columns * target_size[0]
    grid_height = num_rows * target_size[1]
    grid_image = Image.new('RGB', (grid_width, grid_height))
    
    # print(f"Creating image grid: {num_rows} rows x {num_columns} columns = {len(images)} images")
    # print(f"Grid size: {grid_width} x {grid_height}")
    
    for idx, image in enumerate(resized_images):
        row_idx = idx // num_columns
        col_idx = idx % num_columns
        position = (col_idx * target_size[0], row_idx * target_size[1])
        grid_image.paste(image, position)
    
    return grid_image

def make_tmp_video(frames, video_name="input_vid49.mp4", fps=5, temp_dir="temp"):
    # Create temp file and encode for each image
    temp_video_path = os.path.abspath(os.path.join(temp_dir, video_name))
    fps=5
    with imageio.get_writer(
        temp_video_path,
        fps=fps,
        codec="libx264",
        format="FFMPEG",
        pixelformat="yuv420p"
    ) as writer:
        for img in frames:  # img: PIL.Image
            if img.mode != "RGB":
                img = img.convert("RGB")

            frame = np.asarray(img, dtype=np.uint8)
            writer.append_data(frame)
    
    return temp_video_path

def get_answer_from_completion(completion, verbose=False):
    reasoning_content = ""
    answer_content = ""
    is_answering = False
        
    if verbose:
        print("\n" + "=" * 30 + " QVQ Reasoning Process " + "=" * 30)
    
    for chunk in completion:
        if not chunk.choices:
            if hasattr(chunk, 'usage') and verbose:
                print(f"\nAPI Usage: {chunk.usage}")
            pass
        else:
            delta = chunk.choices[0].delta
            
            # Processing reasoning content
            if hasattr(delta, 'reasoning_content') and delta.reasoning_content is not None:
                if verbose:
                    print(delta.reasoning_content, end='', flush=True)
                reasoning_content += delta.reasoning_content
            else:
                # Start outputting final answer
                if delta.content and delta.content != "" and not is_answering:
                    if verbose:
                        print("\n" + "=" * 30 + " QVQ Final Answer " + "=" * 30)
                    is_answering = True
                
                # Processing final answer
                if delta.content:
                    if verbose:
                        print(delta.content, end='', flush=True)
                    answer_content += delta.content
    
    if verbose:
        print("\n" + "=" * 80)
    
    return answer_content

def ping_api(model_name):
    client = OpenAI(
        base_url="http://127.0.0.1:22005/v1",
        api_key="none"
    )

    stream = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": "ping",
            }
        ],
        stream=True,
    )

    for chunk in stream:
        if chunk.choices[0].delta.get("content"):
            break

def call_qvq_api_video(model_name, video_path, prompt, fps=5, temp_dir="temp", verbose=True):
    """
    Call QVQ API for video analysis
    
    Args:
        frames: List of PIL Image objects
        prompt: Analysis prompt text
        temp_dir: Temporary file directory
    
    Returns:
        response: API response text
    """
    if verbose:
        print("Calling api with model:", model_name)
    # Ensure temp directory exists
    os.makedirs(temp_dir, exist_ok=True)
    
    # Save all images and encode
    temp_image_paths = []
    content_list = []
    
    try:

        # Add to content list
        content_list.append({
            "type": "video_url",
            "video_url": {"url": f"file://{video_path}"},
            "fps": fps,
        })
            
        
        # Add text prompt
        content_list.append({"type": "text", "text": prompt})
        
        # Initialize OpenAI client
        client = OpenAI(
            base_url="http://127.0.0.1:22005/v1",
            api_key="none"
        )
        
        if verbose:
            print("Processing calling QVQ API...")
        
        # Create chat completion request
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": content_list,
                }
            ],
            stream=True,
        )
        
        answer_content = get_answer_from_completion(completion)
        
        return answer_content
        
    finally:
        # Clean all temp files
        for temp_path in temp_image_paths:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
        if temp_image_paths and verbose:
            print(f"Cleaned {len(temp_image_paths)} temp files")

def call_qvq_api_multi_images(model_name, frames, prompt, temp_dir="temp", verbose=True):
    """
    Call QVQ API for multi-image analysis
    
    Args:
        frames: List of PIL Image objects
        prompt: Analysis prompt text
        temp_dir: Temporary file directory
    
    Returns:
        response: API response text
    """
    if verbose:
        print("Calling multi image api with model:", model_name)
    # Ensure temp directory exists
    os.makedirs(temp_dir, exist_ok=True)
    
    # Save all images and encode
    temp_image_paths = []
    content_list = []
    
    try:
        if verbose:
            print(f"Processing {len(frames)} keyframes...")
        
        # Create temp file and encode for each image
        for i, frame in enumerate(frames):
            temp_image_path = os.path.join(temp_dir, f"keyframe_{i:03d}.jpg")
            temp_image_paths.append(temp_image_path)
            
            # Save image
            frame.save(temp_image_path, quality=95)
            
            # Encode image
            base64_image = encode_image(temp_image_path)
            
            # Add to content list
            content_list.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
            })
            
            if verbose and (i + 1) % 10 == 0:  # Show progress every 10 images
                print(f"Processed {i + 1}/{len(frames)} keyframes")
        
        # Add text prompt
        content_list.append({"type": "text", "text": prompt})
        
        if verbose:
            print(f"All {len(frames)} keyframes processing completed")
        
        # Initialize OpenAI client
        client = OpenAI(
            # base_url="http://127.0.0.1:22032/v1",
            base_url="http://127.0.0.1:22002/v1",
            # base_url="http://127.0.0.1:22005/v1",
            api_key="none"
        )
        
        if verbose:
            print("Processing calling QVQ API...")
        
        # Create chat completion request
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": content_list,
                }
            ],
            stream=True,
        )
        
        answer_content = get_answer_from_completion(completion)

        if verbose:
            print(answer_content)

        return answer_content
        
    finally:
        # Clean all temp files
        for temp_path in temp_image_paths:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
        if temp_image_paths and verbose:
            print(f"Cleaned {len(temp_image_paths)} temp files")

def call_qvq_api_text(model_name, prompt, verbose=True):
    """
    Call QVQ API for multi-image analysis
    
    Args:
        frames: List of PIL Image objects
        prompt: Analysis prompt text
        temp_dir: Temporary file directory
    
    Returns:
        response: API response text
    """
    if verbose:
        print("Calling multi image api with model:", model_name)
    content_list = []
    
    try:
        # Add text prompt
        content_list.append({"type": "text", "text": prompt})
        
        # Initialize OpenAI client
        client = OpenAI(
            base_url="http://127.0.0.1:22011/v1",
            api_key="none"
        )
        
        if verbose:
            print("Processing calling QVQ API...")
        
        # Create chat completion request
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": content_list,
                }
            ],
            stream=True,
        )
        
        answer_content = get_answer_from_completion(completion)

        if verbose:
            print(answer_content)
    
    except Exception as e:
        if verbose:
            print(e)

    return answer_content

def call_qvq_api_ensure(model_name, input_data, prompt, input_type="images", output_type="plain", temp_dir="temp", verbose=True):
    if verbose:
        print("🔄 Calling QVQ API...")
    
    max_retries = 3
    response = None
    for retry in range(max_retries):
        if verbose:
            print(f"📡 Attempt {retry + 1}/{max_retries}...")
        
        try:
            if input_type == "images":
                response = call_qvq_api_multi_images(model_name, input_data, prompt, temp_dir, verbose=verbose)
            elif input_type == "video":
                if verbose:
                    print(f"Making {len(input_data)} keyframes into video...")
                temp_video_path = make_tmp_video(input_data, fps=5, temp_dir=temp_dir)
                response = call_qvq_api_video(model_name, temp_video_path, prompt, temp_dir=temp_dir, verbose=verbose)
            elif input_type == "video_path":
                if not os.path.exists(input_data):
                    with open("no_vid_list.txt", "a+") as f:
                        f.write(f"{input_data}\n")
                    continue
                response = call_qvq_api_video(model_name, input_data, prompt, temp_dir=temp_dir, verbose=verbose)
            elif input_type == "text":
                response = call_qvq_api_text(model_name, prompt, verbose=verbose)

            if output_type == "structured":
                try:
                    # response = response.split("**Scene**:")[1].strip()
                    # scene_text, camera_text = response.split("**Camera**:")
                    # response = {
                    #     "scene": scene_text.strip(),
                    #     "camera": camera_text.strip()
                    # }

                    response = response.split("**Detailed**:")[1].strip()
                    detail_text, concise_text = response.split("**Concise**:")
                    response = {
                        "detail": detail_text.strip(),
                        "concise": concise_text.strip()
                    }
                except:
                    continue

            if response is not None:
                break
        
        except Exception as e:
            print(f"❌ API call failed: {e}")
            if retry < max_retries - 1:
                print("🔄 Preparing to retry...")
                continue
            else:
                print("❌ Max retries reached, analysis failed")
                return None
    
    return response

def find_continuous_segments(nums: list, segment_len=49, use_remaining_frames=False):
    """
    Returns:
        segments_num  : list[(start_num, end_num_exclusive)]
        segments_idx  : list[(start_idx, end_idx_exclusive)]

    Usage:
        data[start_idx:end_idx]
    """
    nums = sorted(nums)

    segments_num = []
    segments_idx = []

    start_num = nums[0]
    start_idx = 0

    prev_num = nums[0]
    length = 1

    for i in range(1, len(nums)):
        num = nums[i]

        if num == prev_num + 1:
            length += 1
        else:
            start_num = num
            start_idx = i
            length = 1

        if length == segment_len:
            segments_num.append((start_num, num + 1))
            segments_idx.append((start_idx, i + 1))

            # reset for non-overlapping segments
            start_num = num + 1
            start_idx = i + 1
            length = 0

        prev_num = num

    if use_remaining_frames and length < segment_len and length > 10:
        segments_num.append((start_num, num + 1))
        segments_idx.append((start_idx, i + 1))

    return segments_num, segments_idx

def get_title_from_caption(job, use_only_first_caption=True, verbose=False):
    scene_name, scene_dir, split, export_dir = job
    ERR_LIST_PATH = os.path.join(export_dir, "logs", "error_list.txt")
    result_path = os.path.join(export_dir, split, scene_name, "prompts.json")
    os.makedirs(os.path.dirname(ERR_LIST_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    if not os.path.exists(result_path):
        print(f"no caption")
        with open(ERR_LIST_PATH, "a+") as f:
            f.write(f"{split}/{scene_name} no caption\n")
    
    model_name="Qwen3-30B-A3B-Instruct-2507"
    with open(f"prompts/title.txt", "r") as f:
        prompt = f.read()
    with open(result_path, "r") as f:
        caption_data = json.load(f)

    for seg_idx_str, data in caption_data.items():
        if use_only_first_caption and int(seg_idx_str) > 0:
            break
        
        if "prompt_scene" in caption_data[seg_idx_str].keys():
            caption = caption_data[seg_idx_str]["prompt_scene"]
        elif "prompt_camera_with_scene_video_inpainted" in caption_data[seg_idx_str].keys():
            caption = caption_data[seg_idx_str]["prompt_camera_with_scene_video_inpainted"]["detail"]
        elif "prompt_camera_with_scene_video" in caption_data[seg_idx_str].keys():
            caption = caption_data[seg_idx_str]["prompt_camera_with_scene_video"]["detail"]
        else:
            with open(ERR_LIST_PATH, "a+") as f:
                f.write(f"{split}/{scene_name}/{seg_idx_str} no valid caption\n")

        input_prompt = prompt + caption + "\nAnswer:\n"

        try:
            start_time = time.time()
            response = call_qvq_api_ensure(model_name, None, input_prompt, input_type="text", verbose=verbose)
            if response is None:
                raise RuntimeError("No response from api server!")
            
            if verbose:
                print(response)
            result = {}
            result[f"title"] = response
            result[f"metadata_title"] = {
                "model": model_name,
                "analysis_time": datetime.now().isoformat(),
                "inference_time": time.time() - start_time
            }
            
            # Save JSON
            data = {}
            if os.path.exists(result_path) and os.path.getsize(result_path) > 0:
                with open(result_path, "r") as f:
                    data = json.load(f)

            if seg_idx_str in data.keys():
                data[seg_idx_str].update(result)
            else:
                data[seg_idx_str] = result
            
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)

        except Exception as e:
            with open(ERR_LIST_PATH, "a+") as f:
                f.write(f"{split}/{scene_name}/{seg_idx_str} {e}\n")

def separate_scene_text_from_caption(job, verbose=False):
    scene_name, scene_dir, split, export_dir = job
    ERR_LIST_PATH = os.path.join(export_dir, "logs", "error_list.txt")
    result_path = os.path.join(export_dir, split, scene_name, "prompts.json")
    os.makedirs(os.path.dirname(ERR_LIST_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    if not os.path.exists(result_path):
        print(f"no caption")
        with open(ERR_LIST_PATH, "a+") as f:
            f.write(f"{split}/{scene_name} no caption\n")
    
    model_name="Qwen3-30B-A3B-Instruct-2507"
    with open(f"prompts/separate.txt", "r") as f:
        prompt = f.read()
    with open(result_path, "r") as f:
        caption_data = json.load(f)

    for seg_idx_str, data in caption_data.items():
        if "prompt_camera_with_scene_video_inpainted" in caption_data[seg_idx_str].keys():
            caption = caption_data[seg_idx_str]["prompt_camera_with_scene_video_inpainted"]["concise"]
        elif "prompt_camera_with_scene_video" in caption_data[seg_idx_str].keys():
            caption = caption_data[seg_idx_str]["prompt_camera_with_scene_video"]["concise"]
        else:
            with open(ERR_LIST_PATH, "a+") as f:
                f.write(f"{split}/{scene_name}/{seg_idx_str} no valid caption\n")

        input_prompt = prompt + caption + "\n\nAnswer:\n"

        if verbose:
            print("Input:")
            print(caption)

        try:
            start_time = time.time()
            response = call_qvq_api_ensure(model_name, None, input_prompt, input_type="text", output_type="structured", verbose=verbose)
            if response is None:
                raise RuntimeError("No response from api server!")
            
            if verbose:
                print("Output")
                print(response)
            result = {}
            result[f"separate_scene_text"] = response
            result[f"metadata_separate_scene_text"] = {
                "model": model_name,
                "analysis_time": datetime.now().isoformat(),
                "inference_time": time.time() - start_time
            }
            
            # Save JSON
            data = {}
            if os.path.exists(result_path) and os.path.getsize(result_path) > 0:
                with open(result_path, "r") as f:
                    data = json.load(f)

            if seg_idx_str in data.keys():
                data[seg_idx_str].update(result)
            else:
                data[seg_idx_str] = result
            
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)

        except Exception as e:
            with open(ERR_LIST_PATH, "a+") as f:
                f.write(f"{split}/{scene_name}/{seg_idx_str} {e}\n")

def analyze_single_scene(job, segment_length=49, frame_interval=1, temp_dir="temp_qvq", inference_type="all",
                         skip_exist=False, verbose=False, model_handler=None):
    """
    Analyze keyframe folder of a single scene

    inference_type : "camera", "scene", "dynamic"
    skip_exist : skip if result exist
    """
    scene_name, scene_dir, split, export_dir, camera_prompt_path = job
    ERR_LIST_PATH = os.path.join(export_dir, "logs", "error_list.txt")
    result_path = os.path.join(export_dir, split, scene_name, "prompts.json")
    os.makedirs(os.path.dirname(ERR_LIST_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    # if inference_type == "camera_with_scene_video_inpainted":
    #     images_dir = os.path.join(scene_dir, "da3/inpainted/input_images")
    # else:
    images_dir = os.path.join(scene_dir, "images")
    if not os.path.exists(images_dir):
        images_dir = os.path.join(scene_dir, "rgb")
    if not os.path.exists(images_dir):
        print("no img")
        return

    temp_dir = os.path.join(temp_dir, split, scene_name)
    
    if skip_exist and os.path.exists(result_path) and os.path.getsize(result_path) > 0:
        with open(result_path, "r") as f:
            data = json.load(f)

        if data["prompt"] != "":
            print("skipping ", scene_name)
            return "skip"
        
    if verbose:
        print("[Scene Analysis] Processing", split, scene_name)
    
    image_paths = load_keyframe_paths(images_dir)
    

    tasks = []
    if inference_type in ["camera_with_scene_video", "camera_with_scene_video_inpainted"]:
        # Use existing indices
        all_captions = None
        if os.path.exists(camera_prompt_path):
            with open(camera_prompt_path) as f:
                all_captions = json.load(f)
            if len(all_captions.keys()) == 0:
                raise RuntimeError("No camera caption found at", camera_prompt_path)
            for seg_idx_str in all_captions.keys():
                seg_idx = int(seg_idx_str)
                seg_start_idx, seg_end_idx = all_captions[seg_idx_str]["frame_idx"]
                tasks.append((split, scene_name, seg_idx, seg_start_idx, seg_end_idx))
        else:
            raise RuntimeError("No camera caption file found at", camera_prompt_path)
    else:
        # Find continuous image sequence and Add task (이미지 기준 연속된 sequence 찾아서 task 추가)
        image_filenames = sorted(os.listdir(images_dir))
        if image_filenames[0].startswith("frame"):
            img_num_list = [int(img_filename.split('_')[1].split('.')[0]) for img_filename in image_filenames]
        else:
            img_num_list = [int(img_filename.split('.')[0]) for img_filename in image_filenames]
        _, segments_idx = find_continuous_segments(img_num_list, segment_len=segment_length, use_remaining_frames=True)

        # Add subtasks
        for seg_idx, (seg_start_idx, seg_end_idx) in enumerate(segments_idx):
            # Skip first frame
            # if seg_idx == 0 and seg_start_idx == 0:
            #     continue
            tasks.append((split, scene_name, seg_idx, seg_start_idx, seg_end_idx))


    # Execute
    for split, scene_name, seg_idx, seg_start_idx, seg_end_idx in tasks:
        seg_idx_str = str(seg_idx)
    
        try:
            if verbose:
                print("🔄 Loading keyframes...")
            frames, frame_info = load_keyframes(image_paths[seg_start_idx:seg_end_idx], frame_interval=frame_interval)
            
            # Model, input specification
            result = {}
            input_type = "images"
            input_data = frames
            # input_type = "video"
            # input_data = f"{scene_dir}/49frame_vids/vid_49frame_{seg_start_idx:04d}_{seg_end_idx:04d}.mp4"

            model_name = "Qwen/Qwen3-VL-30B-A3B-Instruct"
            if inference_type == "camera_without_tag":
                # model_name = "Qwen/qwen2.5-vl-72b-cam-motion"
                # model_name = "Qwen/qwen2.5-vl-7b-cam-motion"
                pass
            elif inference_type in ["scene", "dynamic", "camera_with_scene_video","camera_with_scene_video_inpainted"]:
                pass
            else:
                raise ValueError(f"Invalid inference type: {inference_type}")
            
            # Prompt
            if inference_type in ["camera_with_scene_video", "camera_with_scene_video_inpainted"]:
                with open("prompts/relationship+video.json", "r") as f:
                    prompt_dict = json.load(f)
                camera_caption = all_captions[seg_idx_str]["prompt_camera"]
                prompt = prompt_dict['context'] + prompt_dict['instruction'] + prompt_dict['constraint'] + prompt_dict['format']
                prompt += "\n\nMovement: " + camera_caption
            else:
                with open(f"prompts/{inference_type}.txt", "r") as f:
                    prompt = f.read()
            
            # Call api
            start_time = time.time()
            if model_handler is not None:
                temp_video_path = make_tmp_video(frames, fps=5, temp_dir=temp_dir)
                response = model_handler.inference(temp_video_path, fps=5, prompt=prompt)[0]
            else:
                response = call_qvq_api_ensure(model_name, input_data, prompt, input_type=input_type, output_type="structured", temp_dir=temp_dir, verbose=verbose)
            
            if response is None:
                raise RuntimeError("No response from api server!")
            
            if verbose:
                print(response)
            
            result[f"prompt_{inference_type}"] = response
            result[f"metadata_{inference_type}"] = {
                "total_frames_used": seg_end_idx - seg_start_idx,
                "model": model_name,
                "analysis_time": datetime.now().isoformat(),
                "inference_time": time.time() - start_time
            }
            
            # Save JSON
            data = {}
            if os.path.exists(result_path) and os.path.getsize(result_path) > 0:
                with open(result_path, "r") as f:
                    data = json.load(f)

            if seg_idx_str in data.keys():
                data[seg_idx_str].update(result)
            else:
                data[seg_idx_str] = result
            
            if "frame_idx" not in data[seg_idx_str].keys():
                data[seg_idx_str]["frame_idx"] = [seg_start_idx, seg_end_idx]
            
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
                
            if verbose:
                print(f"✅ Analysis Completed: {result_path}")
                print("It took", time.time() - start_time)
            
        except Exception as e:
            print(f"❌ Scene analysis failed: {e}")
            with open(ERR_LIST_PATH, "a+") as f:
                f.write(f"{split}/{scene_name}/{seg_idx_str} {e}\n")

    # Remove temp dir
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

def analyze_single_seg(job, temp_dir="temp_qvq", inference_type="all", skip_exist=False, verbose=False, model_handler=None):
    scene_name, scene_dir, split, export_dir = job
    ERR_LIST_PATH = os.path.join(export_dir, "logs", "error_list.txt")
    result_path = os.path.join(export_dir, split, scene_name, "prompts.json")
    camera_prompt_path = os.path.join(camera_prompt_dir, "prompts.json")
    os.makedirs(os.path.dirname(ERR_LIST_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    images_dir = os.path.join(scene_dir, "images")
    if not os.path.exists(images_dir):
        images_dir = os.path.join(scene_dir, "rgb")
    if not os.path.exists(images_dir):
        print("no img")
        return

    temp_dir = os.path.join(temp_dir, split, scene_name)
    
    if skip_exist and os.path.exists(result_path) and os.path.getsize(result_path) > 0:
        with open(result_path, "r") as f:
            data = json.load(f)

        if data["prompt"] != "":
            print("skipping ", scene_name)
            return "skip"
        
    if verbose:
        print("[Scene Analysis] Processing", split, scene_name)
    
    image_paths = load_keyframe_paths(images_dir)

    with open(f"{scene_dir}/prompts.json") as f:
        all_captions = json.load(f)

    # Find continuous image sequence and Add task (이미지 기준 연속된 sequence 찾아서 task 추가)
    image_filenames = sorted(os.listdir(images_dir))
    if image_filenames[0].startswith("frame"):
        img_num_list = [int(img_filename.split('_')[1].split('.')[0]) for img_filename in image_filenames]
    else:
        img_num_list = [int(img_filename.split('.')[0]) for img_filename in image_filenames]
    _, segments_idx = find_continuous_segments(img_num_list, use_remaining_frames=True)

    tasks = []
    for seg_idx, (seg_start_idx, seg_end_idx) in enumerate(segments_idx):
        tasks.append((split, scene_name, seg_idx, seg_start_idx, seg_end_idx))

    for split, scene_name, seg_idx, seg_start_idx, seg_end_idx in tasks:
        seg_idx_str = str(seg_idx)

        if seg_idx_str in all_captions.keys():
            cap_start_idx, cap_end_idx = all_captions[seg_idx_str]["frame_idx"]
            if cap_start_idx != seg_start_idx or cap_end_idx != seg_end_idx:
                print("segment index error")
                with open(ERR_LIST_PATH, "a+") as f:
                    f.write(f"{split}/{scene_name}/{seg_idx_str} index mismatch seg:{seg_start_idx}:{seg_end_idx} cap:{cap_start_idx}:{cap_end_idx}\n")
                continue
    
        # if inference_type not in ["scene", "dynamic", "camera_with_scene_video"]:
        if verbose:
            print("🔄 Loading keyframes...")
        frames, frame_info = load_keyframes(image_paths[seg_start_idx:seg_end_idx])
        
        # Model, input specification
        input_type = "images"
        model_name = "Qwen/Qwen3-VL-30B-A3B-Instruct"
        input_data = frames
        # Prompt
        if inference_type == "camera_with_scene_video":
            with open("prompts/relationship+video.json", "r") as f:
                prompt_dict = json.load(f)
            camera_caption = all_captions[seg_idx_str]["prompt_camera"]
            prompt = prompt_dict['context'] + prompt_dict['instruction'] + prompt_dict['constraint'] + prompt_dict['format']
            prompt += "\n\nMovement: " + camera_caption
        elif inference_type == "camera_with_scene_video_inpainted":
            with open("prompts/relationship+video.json", "r") as f:
                prompt_dict = json.load(f)
            camera_caption = all_captions[seg_idx_str]["prompt_camera"]
            prompt = prompt_dict['context'] + prompt_dict['instruction'] + prompt_dict['constraint'] + prompt_dict['format']
            prompt += "\n\nMovement: " + camera_caption
        else:
            with open(f"prompts/{inference_type}.txt", "r") as f:
                prompt = f.read()
        
        # Call api
        start_time = time.time()
        if model_handler is not None:
            temp_video_path = make_tmp_video(frames, fps=5, temp_dir=temp_dir)
            response = model_handler.inference(temp_video_path, fps=5, prompt=prompt)[0]
        else:
            response = call_qvq_api_ensure(model_name, input_data, prompt, input_type=input_type, temp_dir=temp_dir, verbose=verbose)
        
        if response is None:
            raise RuntimeError("No response from api server!")
        
        if verbose:
            print(response)
            
        if verbose:
            print(f"✅ Analysis Completed: {result_path}")
            print("It took", time.time() - start_time)
        
        return

    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

def _worker(task, **worker_kwargs):
    split, scene_name = task

    root_dir = worker_kwargs["root_dir"]
    export_dir = worker_kwargs["export_dir"]
    temp_dir = worker_kwargs["temp_dir"]
    seg_len = worker_kwargs["seg_len"]
    frame_interval = worker_kwargs["frame_interval"]
    task_type = worker_kwargs["task_type"]
    camera_prompt_root_dir = worker_kwargs.get("camera_prompt_root_dir")
    model_handler = worker_kwargs.get("model_handler")
    camera_prompt_filename = worker_kwargs.get("camera_prompt_filename")
    verbose = worker_kwargs.get("verbose", False)


    scene_dir = os.path.join(root_dir, split, scene_name)
    camera_prompt_path = os.path.join(camera_prompt_root_dir, split, scene_name, camera_prompt_filename)

    if task_type == "title":
        job = (scene_name, scene_dir, split, export_dir)
        get_title_from_caption(
            job, verbose=verbose
        )
    elif task_type == "separate_scene_text":
        job = (scene_name, scene_dir, split, export_dir)
        separate_scene_text_from_caption(
            job, verbose=verbose
        )
    else:
        job = (scene_name, scene_dir, split, export_dir, camera_prompt_path)
        analyze_single_scene(
            job,
            segment_length=seg_len,
            frame_interval=frame_interval,
            temp_dir=temp_dir,
            inference_type=task_type,
            model_handler=model_handler,
            verbose=verbose,
        )

def ensure_worker(task, **worker_kwargs):
    split, scene_name = task
    try:
        _worker(task, **worker_kwargs)
    except Exception as e:
        with open("error_list.txt", "a+") as f:
            f.write(f"{split}/{scene_name} scene analysis failed {e}\n")

def run_parallel(tasks, num_workers=2, **worker_kwargs):
    print("Registered Tasks :", len(tasks))

    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        futures = {ex.submit(ensure_worker, task, **worker_kwargs): task for task in tasks}

        pbar = tqdm(total=len(futures))
        try:
            pending = set(futures)
            while pending:
                done, pending = wait(pending, timeout=0.2, return_when=FIRST_COMPLETED)
                for fut in done:
                    fut.result()
                    pbar.update(1)
        except KeyboardInterrupt:
            for fut in futures:
                fut.cancel()
            ex.shutdown(wait=False, cancel_futures=True)
            raise
        finally:
            pbar.close()

def main():
    parser = argparse.ArgumentParser(description="Call QVQ-max API using keyframes for dynamic object analysis")
    parser.add_argument("--root_dir", required=True, 
                        help="Scene directory path")
    parser.add_argument("--export_dir", default=None, 
                        help="Export directory")
    parser.add_argument("--camera_prompt_root_dir", default=None, 
                        help="Export directory")
    parser.add_argument("--camera_prompt_filename", default="prompts.json", 
                        help="Export directory")
    parser.add_argument("--seg_len", type=int, default=49,
                        help="Segment length (default: 49)")
    parser.add_argument("--frame_interval", type=int, default=1,
                        help="Input frame interval (default: 1)")
    parser.add_argument("--max_frames", type=int, default=49,
                        help="Max frames to use (default: 49)")
    parser.add_argument("--temp_dir", default="temp_qvq",
                        help="Temporary file directory (default: temp_qvq)")
    parser.add_argument("--run_name", type=str, default="prompt_inference")
    parser.add_argument("--load_model", action="store_true")
    parser.add_argument("--splits", nargs='+')
    parser.add_argument("--task_type", type=str, default="")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--single", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if not args.task_type in ["camera",
                            "scene",
                            "dynamic",
                            "camera_with_scene_video",
                            "camera_with_scene_video_inpainted",
                            "title",
                            "separate_scene_text"
                        ]:
        raise ValueError("Invalid task type.", args.task_type)
    
    print(f"🖼️ Max Frames: {args.max_frames}")
    print("Task", args.task_type)

    # ROOT_DIR = "/data1/cympyc1785/SceneData/DL3DV/scenes"
    # root_dir = "/data1/cympyc1785/SceneData/DynamicVerse/scenes"
    root_dir = args.root_dir
    if args.export_dir is None:
        args.export_dir = root_dir
    if args.camera_prompt_root_dir is None:
        args.camera_prompt_root_dir = root_dir

    if args.single:
        split = os.path.basename(os.path.dirname(root_dir))
        scene_name = os.path.basename(root_dir)
        analyze_single_scene(
            (scene_name, root_dir, split, args.export_dir),
            segment_length=49,
            temp_dir=args.temp_dir,
            inference_type=args.task_type,
            model_handler=None,
            verbose=True,
        )
        exit()
    
    # ===== Split Settings =====
    if args.splits is None:
        splits = ["1K", "2K", "3K", "4K", "5K", "6K", "7K"]
        # splits = ["DAVIS", "MOSE", "MVS-Synth", "SAV", "VOST", "dynamic_replica", "spring", "youtube_vis", "uvo"]
        # splits = [f"dynpose-{i:04d}" for i in range(50, 90)]
    elif args.splits[0] == "DL3DV":
        splits = ["1K", "2K", "3K", "4K", "5K", "6K", "7K"]
    elif args.splits[0] == "DynamicVerse":
        splits = ["DAVIS", "MOSE", "MVS-Synth", "SAV", "VOST", "dynamic_replica", "spring", "youtube_vis", "uvo"]
    elif args.splits[0] == "dynpose-100k":
        splits = [f"dynpose-{i:04d}" for i in range(0, 90)]
    else:
        splits = args.splits
        
    print("Splits", splits)

    model_handler = None
    if args.load_model:
        model_handler = QwenModelHandler("qwen2.5-vl-72b-cam-motion")

    # Add all tasks
    tasks = []
    ckpt = None
    if args.ckpt is not None:
        ckpt = args.ckpt.split("/")
        is_ckpt_reached = False
    for split in splits:
        split_root_dir = os.path.join(root_dir, split)
        for scene_name in sorted(os.listdir(split_root_dir))[15:]:
            if args.ckpt is None or is_ckpt_reached:
                tasks.append((split, scene_name))
            elif split == ckpt[0] and scene_name == ckpt[1]:
                is_ckpt_reached = True
                tasks.append((split, scene_name))
            else:
                continue

    # # Rerun split Errors
    # tasks = []
    # with open("retry_list.txt", "r") as f:
    #     err_list_lines = f.readlines()
    # for err_line in tqdm(err_list_lines):
    #     split, scene_name, seg_idx = err_line.split("/")
    #     tasks.append((split, scene_name))

    total_start_time = time.time()

    if not args.debug:
        # Run Parallel
        run_parallel(
            tasks,
            num_workers=4,
            root_dir=root_dir,
            export_dir=args.export_dir,
            camera_prompt_root_dir=args.camera_prompt_root_dir,
            camera_prompt_filename=args.camera_prompt_filename,
            temp_dir=args.temp_dir,
            seg_len=args.seg_len,
            frame_interval=args.frame_interval,
            task_type=args.task_type,
            model_handler=model_handler,
            verbose=False,
        )
    else:
        # Run for Test
        for task in tasks[:5]:
            _worker(
                task,
                root_dir=root_dir,
                export_dir=args.export_dir,
                camera_prompt_root_dir=args.camera_prompt_root_dir,
                camera_prompt_filename=args.camera_prompt_filename,
                temp_dir=args.temp_dir,
                seg_len=args.seg_len,
                frame_interval=args.frame_interval,
                task_type=args.task_type,
                model_handler=model_handler,
                verbose=True,
            )
    
    with open("time.log", "a+") as f:
        f.write(f"{args.task_type} {time.time() - total_start_time}\n")

if __name__ == "__main__":
    main()