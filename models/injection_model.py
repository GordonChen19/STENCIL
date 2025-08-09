from diffusers import StableDiffusion3Pipeline
import torch
from time import datetime

def draft_image(prompt, model_id = "stabilityai/stable-diffusion-3-medium-diffusers", output_folder="output"):

    pipe = StableDiffusion3Pipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")

    image = pipe(
            prompt=prompt,
            negative_prompt="",
            num_inference_steps=50,
            guidance_scale=7.5,
            height = 512,
            width = 512,
        ).images[0]
    

    image_path = f"{output_folder}/draft_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    image.save(image_path)
    return image_path


draft_image("A beautiful landscape with mountains and a river")