# from models.base_model import BaseModel
from helper import prepare_img

# from vlm.extraction_chain import extraction_chain
# from vlm.data_models import AugmentedPrompt, Image
# from vlm.prompt_template import augmented_prompt_template, caption_template
import os
# from PIL import Image

import torch


from models.base_model import base_model
from models.support_model import support_model

from null_text_inversion import NullInversion

from helper import inference, prepare_img

def main(prompt, image_filepath=None):

    reference_folder = "references"
    output_folder = "output"
    os.makedirs(reference_folder, exist_ok = True)
    os.makedirs(output_folder, exist_ok = True)

    
    #######################Generate Captions#############################
    reference_images = []

    # for img_name in os.listdir(reference_folder):
    #     reference_images.append(os.path.join(reference_folder, img_name))

    # #Generate Augmented prompt
    # response = extraction_chain(prompt, AugmentedPrompt, augmented_prompt_template, reference_images[0])

    # subject_name = response["subject_name"]
    # augmented_prompt = response["augmented_prompt"]

    # #Generate caption for every reference image and store in captioned_images

    # captioned_images = []
    # for i, ref_image in enumerate(reference_images):
    #     caption = extraction_chain(subject_name, Image, caption_template, reference_images[i])['image_caption']
    #     captioned_images.append([ref_image, caption])

    # del reference_images

    ########################Generate Template Image######################

    subject_name = 'backpack'
    augmented_prompt = f"A {subject_name} next to the Eiffel tower"

    captioned_images = [['references/00.jpg', 'A person wearing a backpack outdoors.']]

    #Use user-provided image if available
    if image_filepath is None:
        image_filepath = support_model.generate_image(augmented_prompt, output_folder="output")

    del support_model
    torch.cuda.empty_cache()

    ########Generate Cross-Attention Masks for each reference image##########

    subject_latents, subject_embeddings, latent_masks = prepare_img(pipe = base_model.pipe, 
                                                                    captioned_images = captioned_images,
                                                                    target_token = subject_name,
                                                                    threshold=0.2, 
                                                                    steps=50)

    ########################Finetune Base Model######################

    _ = base_model.fine_tune(subject_embeddings, subject_latents, latent_masks, iter=100, bsz=4)

    ####################Perform Null-text Optimization#################

    null_inversion = NullInversion(base_model.pipe)

    (image_gt, image_enc), inverted_latent, uncond_embeddings = null_inversion.invert(image_filepath, prompt, offsets=(0,0,0,0), verbose=True)

    negative_prompt_embeds = torch.cat(uncond_embeddings, dim=0).to(base_model.pipe.device)

        
    ####################Perform Inference############################

    final_image = inference(pipe = base_model.pipe,
                            inverted_latent = inverted_latent,
                            prompt = prompt,
                            negative_prompt_embeds = negative_prompt_embeds,
                            early_stop = 3)

    final_image[0].save(os.path.join(output_folder, "final_image.png"))

if __name__ == "__main__":
    image_filepath = None #For Image-2-Image (None otherwise)
    prompt = "A backpack next to the Eiffel tower"
    main(prompt, image_filepath)