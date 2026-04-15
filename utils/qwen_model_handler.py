# The model is trained on 8.0 FPS which we recommend for optimal inference
import os
import imageio
import numpy as np
import torch

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image

from utils.input_handler import ImagesHandler

DEFAULT_MODEL_DIR = "/data1/cympyc1785/data/motion_dataset/DynamicVerse/preprocess/pretrained"

class QwenModelHandler():
    def __init__(self, model_name="qwen2.5-vl-72b-cam-motion", model_dir=DEFAULT_MODEL_DIR):
        self.model = None


        self.setup_model(model_name, model_dir)
    
    def setup_model(self, model_name="qwen2.5-vl-72b-cam-motion", model_dir=DEFAULT_MODEL_DIR):
        if self.model is not None:
            del self.model
            self.model = None
            torch.cuda.empty_cache()
        
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            f"{model_dir}/{model_name}",
            dtype=torch.bfloat16,
            # attn_implementation="flash_attention_2",
            device_map="auto",
        )

        # default processor
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-72B-Instruct")

        return self.model, self.processor

    def inference(self, video_path, fps=5, prompt:str = None):
        input_prompt = "Describe the camera motion in this video."
        if prompt is not None:
            input_prompt = prompt
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": f"file://{video_path}",
                        "fps": fps,
                    },
                    {"type": "text", "text": input_prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )
        inputs = inputs.to("cuda")

        # Inference
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        print(output_text)
        return output_text

if __name__ == "__main__":
    input_frames_dir = "/data1/cympyc1785/SceneData/DL3DV/scenes/2K/001486e9fac2a00bd4ff24096b88ee0cf07521470062dd234c45e3aa04d04539/images"
    temp_dir = "temp_qvq"
    temp_vid_path = os.path.abspath(f"{temp_dir}/input_vid49.mp4")

    frame_paths = ImagesHandler.process(input_frames_dir)[:49]
    fps = 5.0

    with open("prompts/camera.txt", "r") as f:
        prompt = f.read()

    with imageio.get_writer(
        temp_vid_path,
        fps=fps,
        codec="libx264",
        format="FFMPEG",
        pixelformat="yuv420p"
    ) as writer:
        for p in frame_paths:
            img = Image.open(p).convert("RGB")
            frame = np.asarray(img, dtype=np.uint8)
            writer.append_data(frame)
    print("Made 49 frames as video")

    handler = QwenModelHandler()
    handler.inference(temp_vid_path, fps=fps, prompt=prompt)