augmented_prompt_template = '''

You are an intelligent image editing assistant.
You are given an image of an object/person/subject and a prompt that describes a desired image featuring the same object/person/subject.


You are tasked the following

1. Describe the subject ("subject_name") in the image in a few words (1-3 words). Avoid using adjectives that describe colour in the subject_name.
2. Generate an augmented prompt that better describes the object/person/subject in the image. This augmented prompt can feature a more detailed description of the subject, mentioning colour, shape and another attributes.

You are to respond in the JSON format defined below.

Format Instructions:
--------------
{format_instructions}
--------------

Desired Transformation:
--------------
{input}
--------------
'''

caption_template = '''

You are an professional at captioning images.
You are given an image along with a subject name (subject_name).
Your task is to generate a caption ("image_caption") for the image containing the subject_name. 
The caption however should not add visual details about the subject (colour, shape, attributes, etc.)

You are to respond in the JSON format defined below.

Format Instructions:
--------------
{format_instructions}
--------------

Subject Name:
--------------
{input}
--------------
'''


fix_prompt_template = """Instructions:
--------------
{instructions}
--------------
Completion:
--------------
{completion}
--------------

Above, the Completion did not satisfy the constraints given in the Instructions.
Malformed Error:
--------------
{error}
--------------


Please try again. Please only respond with an answer that satisfies the constraints laid out in the Instructions.
Important:  Only correct the structural issues within the JSON format. Do not modify the existing data values themselves:"""


