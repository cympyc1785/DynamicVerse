# python dynamicgen/motion_aware_key_frame_extract.py \
#     --input_root /data1/cympyc1785/data/motion_dataset/DynamicVerse/SceneData/DL3DV/scenes/1K \
#     --output_root /data1/cympyc1785/data/motion_dataset/DynamicVerse/SceneData/DL3DV/keyframes/1K \
#     --flow_model 'unimatch' \
#     --sample_ratio 0.2


# python prompt_generation.py \
#     --scene_root_dir /data1/cympyc1785/data/motion_dataset/DynamicVerse/SceneData/DL3DV/scenes/1K \
#     --keyframe_root_dir /data1/cympyc1785/data/motion_dataset/DynamicVerse/SceneData/DL3DV/keyframes/1K

# python prompt_generation.py --task camera_with_scene_video

# python prompt_generation.py --task scene

# python prompt_generation.py --splits uvo --task camera_with_scene_video_inpainted --ckpt "uvo/-0LoCy0-F9A"

# python prompt_generation.py --splits uvo --task camera_with_scene_video

# python prompt_generation.py --task camera_with_scene_video_inpainted --root_dir "/data1/cympyc1785/data/motion_dataset/DynamicVerse/SceneData/DynamicVerse/scenes/uvo/15LSh_XPils" --single

# python prompt_generation.py --task camera_with_scene_video_inpainted --root_dir "/data1/cympyc1785/SceneData/DynamicVerse/data/dynpose-100k"

# 이거 다음 task
# python prompt_generation.py --task camera_with_scene_video --root_dir "/data1/cympyc1785/SceneData/DynamicVerse/data/dynpose-100k" --ckpt dynpose-0054/0f73a331-9d9e-4cee-96df-bf1ce143784f --splits dynpose-100k

# python prompt_generation.py --splits DAVIS MOSE MVS-Synth SAV VOST dynamic_replica spring youtube_vis uvo --task scene

# python prompt_generation.py --splits DAVIS MOSE MVS-Synth SAV VOST dynamic_replica spring youtube_vis uvo --task dynamic

# notify-mail cympyc1785@gmail.com "[VCAI-Server1] Captioning Stopped" "Your captioning stopped"; kill $(lsof -ti :22002)

# python prompt_generation.py --task title --splits 2K --root_dir "/data1/cympyc1785/SceneData/DL3DV/scenes"

# python prompt_generation.py --task title --root_dir "/data1/cympyc1785/SceneData/DynamicVerse/scenes"

# python prompt_generation.py --task title --root_dir "/data1/cympyc1785/SceneData/DynamicVerse/data/dynpose-100k"

# python prompt_generation.py --task separate_scene_text --root_dir "/data1/cympyc1785/SceneData/DL3DV/scenes" --splits DL3DV

python prompt_generation.py --task separate_scene_text --root_dir "/data1/cympyc1785/SceneData/DynamicVerse/scenes/dynpose-100k" --splits dynpose-100k

# python prompt_generation.py --task separate_scene_text --root_dir "/data1/cympyc1785/SceneData/DynamicVerse/scenes/dynpose-100k" --splits dynpose-100k

notify-mail cympyc1785@gmail.com "[VCAI-Server1] Captioning Stopped" "Your captioning stopped";
