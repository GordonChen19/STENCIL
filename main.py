# from models.base_model import BaseModel
# from models.support_model import SupportModel
from vlm.extraction_chain import extraction_chain
from vlm.data_models import AugmentedPrompt, Image
from vlm.prompt_template import augmented_prompt_template, caption_template
import os
# from PIL import Image




def main(prompt, image_filepath=None):

    reference_folder = "references"
    reference_images = []


    for img_name in os.listdir(reference_folder):
        reference_images.append(os.path.join(reference_folder, img_name))

    #Generate Augmented prompt
    response = extraction_chain(prompt, AugmentedPrompt, augmented_prompt_template, reference_images[0])

    subject_name = response["subject_name"]
    augmented_prompt = response["augmented_prompt"]

    print(subject_name, augmented_prompt)

    #Generate caption for every reference image

    captioned_images = []
    for i, ref_image in enumerate(reference_images):
        caption = extraction_chain(subject_name, Image, caption_template, reference_images[i])['image_caption']
        captioned_images.append([ref_image, caption])

    print(captioned_images)

    return
    #Instantiate the base model
    base_model = BaseModel()

    #Instantiate the support model
    support_model = SupportModel()


    #Generate high-fidelity image
    if image_filepath is None:
        image_filepath = support_model.generate_image(augmented_prompt)
    
    


    


    

   



if __name__ == "__main__":
    image_filepath = None #For Image-2-Image (None otherwise)
    prompt = "A backpack next to the Eiffel tower"
    main(prompt, image_filepath)