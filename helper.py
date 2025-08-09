
import numpy as np

from PIL import Image
from tqdm.notebook import tqdm

import torch
import torch.nn.functional as F
from torchvision import transforms



from IPython.display import display



import matplotlib.pyplot as plt


import os

def display_attn_mask(attn_mask):
    threshold_mask = (attn_mask.cpu().squeeze(0).squeeze(0).numpy()).astype(np.uint8)
    pil_image = Image.fromarray(threshold_mask, mode='L')
    plt.imshow(pil_image, cmap='viridis')
    plt.axis('off')
    plt.show()

def get_image(prompt=None, emb=None, guidance_scale=7.5, negative_prompt_embeds = None, num_inference_steps=50, num_images=1):
    if prompt is not None:
        image = pipe(prompt=prompt, guidance_scale=guidance_scale, num_inference_steps = num_inference_steps, num_images_per_prompt = num_images, negative_prompt_embeds=negative_prompt_embeds).images
    elif emb is not None:
        image = pipe(prompt_embeds=emb, guidance_scale=guidance_scale, num_inference_steps = num_inference_steps, num_images_per_prompt = num_images, negative_prompt_embeds=negative_prompt_embeds).images
    return image


def encode_txt(prompt):
    inp = pipe.tokenizer(prompt,
                    padding="max_length",
                    max_length=pipe.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt").to(pipe.device)
    with torch.no_grad():
        text_embeddings = pipe.text_encoder(inp.input_ids)[0]
    return text_embeddings.to(pipe.device), inp

def encode_img(obj, dim = 512):

    if not isinstance(obj, Image.Image): #If url
        image = load_image(obj)
    else:
        image = obj

    tform = transforms.Compose([transforms.Resize(dim),
                                transforms.CenterCrop(dim),
                                transforms.ToTensor(),
                                transforms.Normalize(mean =0.5, std =0.5)
                            ])
    image = tform(image).unsqueeze(0).to(device = pipe.device)

    with torch.no_grad():
        latent = pipe.vae.encode(image).latent_dist.sample() * pipe.vae.config.scaling_factor

    return latent, image

@torch.no_grad()
def get_mask(latent, target_prompt, inverted_latent=None, start_step=30, threshold=None, steps=100):


    save_cross_attention_map()
    timestep = torch.tensor(100, dtype=torch.long, device=pipe.device)

    noise = torch.randn_like(latent)
    noisy_latents = pipe.scheduler.add_noise(latent, noise, timestep)

    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(target_prompt, pipe.device, 1, True)
    prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

    latent_model_input = torch.cat([latent]*2)
    latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, timestep)

    _ = pipe.unet(
        latent_model_input,
        timestep,
        encoder_hidden_states=prompt_embeds
    ).sample


    avg_attention_map = pipe.attention_maps

    avg_attention_map = (avg_attention_map - avg_attention_map.min()) / (avg_attention_map.max() - avg_attention_map.min())

    gray_map = F.interpolate(
            avg_attention_map, size=(64, 64), mode="bilinear", align_corners=False
        ).squeeze().cpu().numpy()
    plt.imshow(gray_map, cmap='gray_r')  # 'gray_r' inverts grayscale (darker = larger values)
    plt.axis('off')  # Hide axis for better visualization
    plt.show()
    if threshold is not None:
        threshold_mask = (avg_attention_map > threshold).to(torch.float32)
    else:
        threshold_mask = avg_attention_map

    attention_map_rescaled = F.interpolate(
            threshold_mask, size=(64, 64), mode="bilinear", align_corners=False
        )


    return threshold_mask, attention_map_rescaled

@torch.no_grad()
def prepare_img(file_path, images, threshold=None, steps=100):

    subject_latents = []
    subject_embeddings = []
    latent_masks = []

    for image in images:
        file_name = image['file']
        target_token = image['response']['subject_name']
        subject_caption = image['response']['subject_caption']
        image_caption = image['response']['image_caption']
        print(image_caption)

        target_word_id = pipe.tokenizer(target_token, return_tensors="pt").input_ids[0][1]  # Get token ID

        image_latent, image_tensor = encode_img(os.path.join(file_path,file_name))


        subject_latents.append(image_latent)

        #training class token

        # subject_caption = subject_caption.replace(target_token, f'sks {target_token}')

        subject_emb, _ = encode_txt(subject_caption)
        subject_embeddings.extend([subject_emb])

        _, token_sequence = encode_txt(image_caption)

        for j, token in enumerate(token_sequence['input_ids'][0]):
            if token == target_word_id:
                pipe.target_token_idx=j
                break


        attn_mask, latent_mask = get_mask(image_latent, image_caption, threshold=threshold, steps=steps)

            # img = Image.open(os.path.join(file_path,file_name))
            # img = ImageOps.exif_transpose(img)
            # image_pil = img.convert("RGB")

            # tform = transforms.Compose([transforms.Resize(512),
            #                 transforms.CenterCrop(512)
            #             ])

            # image_pil = tform(image_pil)

            # attn_mask = get_masks_and_boxes(image_pil, target)

            # attn_mask = torch.from_numpy(attn_mask) #512x512

            # latent_mask = F.interpolate(
            #    attn_mask.unsqueeze(0).unsqueeze(0), size=(64, 64), mode="bilinear", align_corners=False
            # ).to(pipe.device)
            # clear_output()

        tform = transforms.Compose([transforms.Resize(512),
                            transforms.CenterCrop(512)
                        ])
        display(tform(Image.open(os.path.join(file_path,file_name))))
        display_attn_mask(attn_mask*255)
        latent_masks.append(latent_mask)



    subject_embeddings = torch.cat(subject_embeddings, dim = 0).clone().detach().requires_grad_(True)
    subject_latents = torch.cat(subject_latents, dim = 0)
    latent_masks = torch.cat(latent_masks, dim = 0)
    return subject_latents, subject_embeddings, latent_masks

@torch.no_grad()
def forward(latent, timestep, prompt_embeds, guidance_scale=7.5):

    latent_model_input = torch.cat([latent]*2)
    latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, timestep)

    noise_pred_cond = pipe.unet(
        latent_model_input,
        timestep,
        encoder_hidden_states=prompt_embeds
    ).sample

    #Perform guidance
    noise_pred_uncond, noise_pred_cond = noise_pred_cond.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (
        noise_pred_cond - noise_pred_uncond
    )

    latent = pipe.scheduler.step(noise_pred, timestep, latent).prev_sample
    return latent

@torch.no_grad()
def inference(
    inverted_latent,
    prompt="",
    guidance_scale=7.5,
    steps=50,
    negative_prompt_embeds = None,
    early_stop = 5
):

    pipe.scheduler.set_timesteps(steps)
    timesteps = pipe.scheduler.timesteps

    latent = inverted_latent
    for i, t in tqdm(enumerate(timesteps), total=len(timesteps)):
        print(i)
        if i<early_stop:
            prompt_embeds, neg_prompt_embeds = pipe.encode_prompt(prompt, pipe.device, 1, True, negative_prompt_embeds = negative_prompt_embeds[i].unsqueeze(0))
            prompt_embeds = torch.cat([neg_prompt_embeds, prompt_embeds])
        else:
            prompt_embeds, neg_prompt_embeds = pipe.encode_prompt(prompt, pipe.device, 1, True)
            prompt_embeds = torch.cat([neg_prompt_embeds, prompt_embeds])

        latent = forward(latent, t, prompt_embeds, guidance_scale)


    # scale and decode the image latents with vae
    latent = latent / pipe.vae.config.scaling_factor
    image = pipe.vae.decode(latent).sample

    return pipe.image_processor.postprocess(image, output_type='pil')
