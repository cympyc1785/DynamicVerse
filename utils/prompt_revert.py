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
    parser.add_argument("scene_root_dir", help="Scene directory path")
    args = parser.parse_args()

    scene_root_dir = args.scene_root_dir
    
    image_folders = find_image_folders(scene_root_dir)
    total_start_time = time.time()

    fail=[]
    for folder in tqdm(image_folders):
        scene_path = os.path.dirname(os.path.normpath(folder))
        scene_name = os.path.basename(os.path.normpath(scene_path))

        result_filename = f"prompt.json"
        result_path = os.path.join(scene_path, result_filename)
        bak_path = result_path + ".bak"

        if os.path.exists(bak_path):
            if os.path.exists(result_path):
                os.remove(result_path)
            shutil.move(bak_path, result_path)
        

if __name__ == "__main__":
    main()