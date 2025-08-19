from diffusers import StableDiffusion3Pipeline
import torch
from time import datetime

class SupportModel():
    def __init__(self, model_id="stabilityai/stable-diffusion-3-medium-diffusers"):
        self.model_id = model_id
        self.pipe = StableDiffusion3Pipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        self.pipe = self.pipe.to("cuda")
    
    def generate_image(self, prompt, output_folder = "output"):
        image = self.pipe(
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


#Instantiate the support model
support_model = SupportModel()