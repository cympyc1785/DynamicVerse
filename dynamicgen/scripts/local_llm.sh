# CUDA_VISIBLE_DEVICES=5
python -m vllm.entrypoints.openai.api_server \
  --model /data1/cympyc1785/data/motion_dataset/DynamicVerse/preprocess/pretrained/Qwen3-30B-A3B-Instruct-2507 \
  --served-model-name Qwen3-30B-A3B-Instruct-2507 \
  --tensor-parallel-size 1 \
  --mm-encoder-tp-mode data \
  --host 0.0.0.0 \
  --port 22011 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.7 \
  --quantization fp8 \
  --distributed-executor-backend mp \
  --allowed-local-media-path /data1/cympyc1785/data/motion_dataset/DynamicVerse/temp_qvq