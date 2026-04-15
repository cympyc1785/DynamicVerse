
CUDA_VISIBLE_DEVICES=1
python stage2_sa2va.py \
    --images \
    --image_folder /data1/cympyc1785/data/motion_dataset/DynamicVerse/data/demo/scene1/rgb \
    --video_fps 10\
    dummy_input \
    /data1/cympyc1785/data/motion_dataset/DynamicVerse/data/demo/scene1/analysis/dynamic_objects_scene1.json \
    /data1/cympyc1785/data/motion_dataset/DynamicVerse/data/demo/scene1/segmentation \
