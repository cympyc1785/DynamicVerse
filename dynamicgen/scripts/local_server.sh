export VLLM_USE_V1=0

python -m vllm.entrypoints.openai.api_server \
  --model /data1/cympyc1785/data/motion_dataset/DynamicVerse/preprocess/pretrained/Qwen3-VL-30B-A3B-Instruct \
  --served-model-name Qwen/Qwen3-VL-30B-A3B-Instruct \
  --tensor-parallel-size 1 \
  --mm-encoder-tp-mode data \
  --host 0.0.0.0 \
  --port 22002 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.70 \
  --quantization fp8 \
  --distributed-executor-backend mp \
  --allowed-local-media-path /data1/cympyc1785/data/motion_dataset/DynamicVerse/temp_qvq \
  # --allowed-local-media-path /data1/cympyc1785/SceneData/DL3DV/scenes
  
#   --enable-expert-parallel 

# CUDA_VISIBLE_DEVICES=0,2
# python -m vllm.entrypoints.openai.api_server \
#   --model /data1/cympyc1785/data/motion_dataset/DynamicVerse/preprocess/pretrained/qwen2.5-vl-72b-cam-motion \
#   --served-model-name Qwen/qwen2.5-vl-72b-cam-motion \
#   --tensor-parallel-size 2 \
#   --mm-encoder-tp-mode data \
#   --host 0.0.0.0 \
#   --port 22002 \
#   --dtype bfloat16 \
#   --gpu-memory-utilization 0.9 \
#   --quantization fp8 \
#   --distributed-executor-backend mp \
#   --allowed-local-media-path /data1/cympyc1785/data/motion_dataset/DynamicVerse/temp_qvq \
#   # --max-model-len 8192
#   # --chat-template-content-format openai

# CUDA_VISIBLE_DEVICES=2
# python -m vllm.entrypoints.openai.api_server \
#   --model /data1/cympyc1785/data/motion_dataset/DynamicVerse/preprocess/pretrained/qwen2.5-vl-7b-cam-motion \
#   --served-model-name Qwen/qwen2.5-vl-7b-cam-motion \
#   --tensor-parallel-size 1 \
#   --mm-encoder-tp-mode data \
#   --host 0.0.0.0 \
#   --port 22002 \
#   --dtype bfloat16 \
#   --gpu-memory-utilization 0.7 \
#   --quantization fp8 \
#   --distributed-executor-backend mp \
#   --allowed-local-media-path /data1/cympyc1785/data/motion_dataset/DynamicVerse/temp_qvq

# python -m vllm.entrypoints.openai.api_server \
#   --model /data1/cympyc1785/data/motion_dataset/DynamicVerse/preprocess/pretrained/sharegpt4video-8b \
#   --served-model-name sharegpt4video-8b \
#   --tensor-parallel-size 1 \
#   --mm-encoder-tp-mode data \
#   --host 0.0.0.0 \
#   --port 22002 \
#   --dtype bfloat16 \
#   --gpu-memory-utilization 0.70 \
#   --quantization fp8 \
#   --distributed-executor-backend mp \
#   --trust-remote-code
