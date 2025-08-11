from pydantic import BaseModel, Field

class AugmentedPrompt(BaseModel):
    subject_name: str = Field(description = '''subject_name is a description of the object/person/animal of the image (1-2 words).''', examples=["dog", "cat", "car", "dog plushie", "toy car", "cartoon devil"]) 
    augmented_prompt: str = Field(description = '''augmented_prompt''')

class Image(BaseModel):
    image_caption: str = Field(description = '''image_caption: a concise caption of the image. 
                               The image_caption should contain the subject_name.''',
                               examples = ["A dog plushie on a bench", "A toy car on a living room table", "A cartoon devil flexing with a white background"])

