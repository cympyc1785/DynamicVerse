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

def encode_image(image_path):
    """Encode image file to base64 format"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def load_keyframes_from_directory(keyframes_dir, max_frames=64):
    """
    Load previously extracted keyframes from directory
    
    Args:
        keyframes_dir: Keyframes directory path
        max_frames: Max frames to use, to avoid grid becoming too large
    
    Returns:
        frames: List of PIL Image objects
        frame_info: List of frame information
    """
    print(f"Processing loading keyframes from directory: {keyframes_dir}")
    
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
    
    if not image_paths:
        raise ValueError(f"No image files found in directory {keyframes_dir}")
    
    print(f"Found {len(image_paths)} image files")
    
    # If too many images, perform uniform sampling
    if len(image_paths) > max_frames:
        print(f"Image count ({len(image_paths)}) exceeds max limit ({max_frames}), performing slicing")
        image_paths = image_paths[:max_frames]
        # print(f"Image count ({len(image_paths)}) exceeds max limit ({max_frames}), performing uniform sampling")
        # indices = np.linspace(0, len(image_paths) - 1, max_frames, dtype=int)
        # image_paths = [image_paths[i] for i in indices]
        # print(f"Using {len(image_paths)} images after sampling")
    
    # Load images
    frames = []
    frame_info = []
    
    for i, image_path in enumerate(image_paths):
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
    
    print(f"Successfully loaded {len(frames)} keyframes")
    return frames, frame_info

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
    
    print(f"Creating image grid: {num_rows} rows x {num_columns} columns = {len(images)} images")
    print(f"Grid size: {grid_width} x {grid_height}")
    
    for idx, image in enumerate(resized_images):
        row_idx = idx // num_columns
        col_idx = idx % num_columns
        position = (col_idx * target_size[0], row_idx * target_size[1])
        grid_image.paste(image, position)
    
    return grid_image

def call_qvq_api_multi_images(frames, prompt, temp_dir="temp"):
    """
    Call QVQ API for multi-image analysis
    
    Args:
        frames: List of PIL Image objects
        prompt: Analysis prompt text
        temp_dir: Temporary file directory
    
    Returns:
        response: API response text
    """
    # Ensure temp directory exists
    os.makedirs(temp_dir, exist_ok=True)
    
    # Save all images and encode
    temp_image_paths = []
    content_list = []
    
    try:
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
            
            if (i + 1) % 10 == 0:  # Show progress every 10 images
                print(f"Processed {i + 1}/{len(frames)} keyframes")
        
        # Add text prompt
        content_list.append({"type": "text", "text": prompt})
        
        print(f"All {len(frames)} keyframes processing completed")
        
        # Initialize OpenAI client
        client = OpenAI(
            base_url="http://127.0.0.1:22002/v1",
            api_key="none"
        )
        
        print("Processing calling QVQ API...")
        
        # Create chat completion request
        completion = client.chat.completions.create(
            model="Qwen/Qwen3-VL-30B-A3B-Instruct",
            messages=[
                {
                    "role": "user",
                    "content": content_list,
                }
            ],
            stream=True,
        )
        
        reasoning_content = ""
        answer_content = ""
        is_answering = False
        
        print("\n" + "=" * 30 + " QVQ Reasoning Process " + "=" * 30)
        
        for chunk in completion:
            if not chunk.choices:
                if hasattr(chunk, 'usage'):
                    print(f"\nAPI Usage: {chunk.usage}")
            else:
                delta = chunk.choices[0].delta
                
                # Processing reasoning content
                if hasattr(delta, 'reasoning_content') and delta.reasoning_content is not None:
                    print(delta.reasoning_content, end='', flush=True)
                    reasoning_content += delta.reasoning_content
                else:
                    # Start outputting final answer
                    if delta.content and delta.content != "" and not is_answering:
                        print("\n" + "=" * 30 + " QVQ Final Answer " + "=" * 30)
                        is_answering = True
                    
                    # Processing final answer
                    if delta.content:
                        print(delta.content, end='', flush=True)
                        answer_content += delta.content
        
        print("\n" + "=" * 80)
        return answer_content
        
    finally:
        # Clean all temp files
        for temp_path in temp_image_paths:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
        if temp_image_paths:
            print(f"Cleaned {len(temp_image_paths)} temp files")

def analyze_single_scene(scene_name, scene_path, keyframe_dir, prompt, max_frames=64, temp_dir="temp_qvq"):
    """
    Analyze keyframe folder of a single scene
    """
    print(f"\n🔄 Analyzing scene: {scene_name}")
    print(f"📁 Keyframe Directory: {keyframe_dir}")
    start_time = time.time()
    try:
        # 1. Loading keyframes
        print("🔄 Loading keyframes...")
        frames, frame_info = load_keyframes_from_directory(keyframe_dir, max_frames=max_frames)
        
        # 2. Call QVQ API
        print("🔄 Calling QVQ API...")

        max_retries = 3
        result = None
        
        for retry in range(max_retries):
            print(f"📡 Attempt {retry + 1}/{max_retries}...")
            
            try:
                response = call_qvq_api_multi_images(frames, prompt, temp_dir)

                if response is not None:
                    break
                        
            except Exception as e:
                print(f"❌ API call failed: {e}")
                if retry < max_retries - 1:
                    print("🔄 Preparing to retry...")
                    continue
                else:
                    print("❌ Max retries reached, analysis failed")
                    return False, None
        
        # 3. Saving results
        print("🔄 Saving analysis results...")

        result = {
            "prompt": response
        }
        
        # Add metadata
        result["metadata"] = {
            "scene_name": scene_name,
            "input_dir": keyframe_dir,
            "total_frames_used": len(frames),
            "analysis_time": datetime.now().isoformat(),
            "inference_time": time.time() - start_time         
        }

        print(f"✅ Analysis Completed")
        print("It took", time.time() - start_time)
        print(result)
        
        return True, result
        
    except Exception as e:
        print(f"❌ Scene analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def find_image_folders(root_dir):
        """Find all scene folders in root directory, these folders should contain a subfolder named 'images'"""
        image_folders = []
        print(f"Searching for scenes with 'images' subfolder in: {root_dir}")
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
                            print(f"Found image folder for scene '{scene_name}': {images_folder}")
                    except OSError as e:
                         print(f"Warning: Cannot read directory {images_folder}: {e}")
        print(f"Total found {len(image_folders)} image folders")
        return sorted(image_folders)

def main():
    parser = argparse.ArgumentParser(description="Call QVQ-max API using keyframes for dynamic object analysis")
    parser.add_argument("--hash", required=True, 
                        help="Scene directory path")
    parser.add_argument("--max_frames", type=int, default=49,
                        help="Max frames to use (default: 49)")
    parser.add_argument("--temp_dir", default="temp_qvq",
                        help="Temporary file directory (default: temp_qvq)")
    args = parser.parse_args()

    scene_root_dir = "/data1/cympyc1785/SceneData/DL3DV/scenes/1K"
    keyframe_root_dir = "/data1/cympyc1785/SceneData/DL3DV/keyframes/1K"

    with open("prompt_2.txt", "r") as f:
        prompt = f.read()
    
    scene_name = args.hash
    scene_path = os.path.join(scene_root_dir, scene_name)
    # keyframe_dir = os.path.join(keyframe_root_dir, scene_name, 'rgb')
    keyframe_dir = os.path.join(scene_path, 'images')
    success, result_path = analyze_single_scene(
        scene_name, scene_path, keyframe_dir, prompt,
        max_frames=args.max_frames, temp_dir=args.temp_dir
    )
        

if __name__ == "__main__":
    main()