# mkdir -p ./preprocess/pretrained

wget https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth -O ./preprocess/pretrained/scaled_offline.pth
# huggingface-cli download lpiccinelli/unidepth-v2old-vitl14 --local-dir ./preprocess/pretrained/unidepth_model
# huggingface-cli download ByteDance/Sa2VA-8B --local-dir ./preprocess/pretrained/Sa2VA-8B
# huggingface-cli download Qwen/Qwen3-VL-235B-A22B-Instruct --local-dir ./preprocess/pretrained/Qwen3-VL-235B-A22B-Instruct
# huggingface-cli download Qwen/Qwen2.5-VL-72B-Instruct --local-dir ./preprocess/pretrained/Qwen2.5-VL-72B-Instruct
huggingface-cli download Qwen/Qwen3-VL-30B-A3B-Instruct --local-dir ./preprocess/pretrained/Qwen3-VL-30B-A3B-Instruct

notify-mail cympyc1785@gmail.com "[VCAI-Server1] DynamicVerse Download Stopped" "Your download stopped"