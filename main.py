from models.base_model import BaseModel
from models.support_model import SupportModel
from vlm.extraction_chain import extraction_chain
from vlm.data_models import Entity
from vlm.prompt_template import prompt_template
import os
from PIL import Image




def main(prompt, image_filepath=None):

    reference_folder = "references"
    reference_images = []


    for img_name in os.listdir(reference_folder):
        reference_images.append(os.path.join(reference_folder, img_name))

    #Generate Augmented prompt
    augmented_prompt = extraction_chain(prompt, Entity, prompt_template, reference_images[0])['augmented_prompt']

    
    #Instantiate the base model
    base_model = BaseModel()

    #Instantiate the support model
    support_model = SupportModel()


    #Generate high-fidelity image
    if image_filepath is None:
        image_filepath = support_model.generate_image(augmented_prompt)
    
    


    


    

   



if __name__ == "__main__":
    image_filepath = None #For Image-2-Image
    prompt = "A backpack next to the Eiffel tower"
    main(prompt, image_filepath)